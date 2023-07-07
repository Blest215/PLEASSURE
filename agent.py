import tensorflow as tf
import numpy as np

from typing import List, Dict

from utils import ExperienceMemory


class ServiceAgent:
    def __init__(
            self, memory_size: int, learning_rate: float, embedding_size: int, embedding_activation, layer_specs,
            epochs: int, patience: int, validation_fold: int, batch_size: int,
            ragged: bool, user_state_size: int, service_state_size: int
    ):
        self.service = None

        # Learning
        self.memory = ExperienceMemory(memory_size)
        self.epochs = epochs
        self.batch_size = batch_size

        # Validation
        self.validation_fold = validation_fold
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience)]

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        # Neural network
        self.user_state_size = user_state_size
        self.service_state_size = service_state_size
        self.embedding_size = embedding_size
        self.embedding_activation = embedding_activation
        self.layer_specs = layer_specs

        self.ragged = ragged

        self.fingerprint_variable = None

        self.model = self.build_model()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mean_squared_error"],
        )

    def set_service(self, service):
        self.service = service

    def build_model(self) -> tf.keras.Model:
        pass

    def align_inputs(self, samples: List[Dict]):
        pass

    def align_outputs(self, samples: List[Dict]):
        pass

    def learn(self, new_experiences: List[dict]):
        for experience in new_experiences:
            self.memory.add(**experience)

        loss_list = []
        satisfaction_error_list = []
        interference_error_list = []

        if self.memory.length() >= self.validation_fold:
            folds = self.memory.split(self.validation_fold)

            for k in range(len(folds)):
                train_set = []
                for i in range(len(folds)):
                    if i != k:
                        train_set += folds[i]
                validation_set = folds[k]

                history = self.model.fit(
                    self.align_inputs(train_set),
                    self.align_outputs(train_set),
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    validation_data=(self.align_inputs(validation_set), self.align_outputs(validation_set)),
                    verbose=0,
                )
                loss_list += history.history["loss"]
                satisfaction_error_list += history.history["satisfaction_mean_squared_error"]
                interference_error_list += history.history["interference_mean_squared_error"]

        return (
            loss_list,
            satisfaction_error_list,
            interference_error_list,
        )

    def prediction(self, samples: List[Dict], training: bool = False) -> List[float]:
        inputs = self.align_inputs(samples)
        satisfaction, interference = self.model(inputs, training=training)
        return tf.squeeze(satisfaction - interference).numpy()

    @property
    def fingerprint(self):
        return self.fingerprint_variable.numpy() if self.fingerprint_variable is not None else None


class IndependentAgent(ServiceAgent):
    def build_model(self):
        # Model
        observation = tf.keras.Input(shape=(self.user_state_size,), dtype=tf.float32, name='observation')
        observation_embedding = tf.keras.layers.Dense(
            units=self.embedding_size, activation=self.embedding_activation
        )(observation)

        x = observation_embedding
        for spec in self.layer_specs:
            x = spec.instantiate()(x)
        satisfaction = tf.keras.layers.Dense(units=1, activation=None, name='satisfaction')(x)
        interference = tf.keras.layers.Dense(units=1, activation=None, name='interference')(x)

        model = tf.keras.Model(
            inputs=[observation],
            outputs=[satisfaction, interference],
        )
        return model

    def align_inputs(self, samples: List[Dict]):
        return {
            "observation": np.array([sample["observation"] for sample in samples]),
        }

    def align_outputs(self, samples: List[Dict]):
        return {
            "satisfaction": np.array([[sample["satisfaction"]] for sample in samples]),
            "interference": np.array([[sample["interference"]] for sample in samples]),
        }


class FingerprintAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_size: int, ragged: bool):
        super(FingerprintAttention, self).__init__()
        self.fingerprint = self.add_weight(
            name="fingerprint", shape=(embedding_size,), trainable=True, dtype=tf.float32,
            initializer=tf.keras.initializers.Ones(),
        )
        self.w = self.add_weight(shape=(embedding_size, embedding_size), trainable=True, dtype=tf.float32)
        self.b = self.add_weight(shape=(embedding_size,), trainable=True, dtype=tf.float32)

        self.scale = tf.sqrt(tf.constant(value=embedding_size, dtype=tf.float32))
        self.ragged = ragged

    def calculate_scores(self, key):
        return key.with_values(
            tf.einsum("ik,kk,k->i", key.flat_values, self.w, self.fingerprint)
        ) if self.ragged else tf.einsum(
            "ijk,k->ij", tf.add(tf.einsum("ijk,kk->ijk", key, self.w), self.b), self.fingerprint
        )

    def call(self, inputs, training=None, **kwargs):
        # inputs[1]: key
        scores = self.calculate_scores(inputs[1])
        weights = tf.expand_dims(tf.nn.softmax(scores / self.scale, axis=-1), axis=-1)
        # inputs[0]: value
        return tf.reduce_sum(tf.multiply(weights, tf.concat([inputs[0], inputs[1]], axis=-1)), axis=-2)


class FingerprintAttentionAgent(ServiceAgent):
    def build_model(self) -> tf.keras.Model:
        # Model
        observation = tf.keras.Input(shape=(self.user_state_size,), dtype=tf.float32, name='observation')
        observation_embedding = tf.keras.layers.Dense(
            units=self.embedding_size, activation=self.embedding_activation,
        )(observation)

        service_fingerprints = tf.keras.Input(
            shape=(None, self.embedding_size), dtype=tf.float32, name='service_fingerprints',
            ragged=self.ragged,
        )
        service_contexts = tf.keras.Input(
            shape=(None, self.service_state_size), dtype=tf.float32, name='service_contexts',
            ragged=self.ragged,
        )
        service_contexts_embedding = tf.keras.layers.Dense(
            units=self.embedding_size, activation=self.embedding_activation,
            use_bias=False,
        )(service_contexts)

        attention_layer = FingerprintAttention(self.embedding_size, self.ragged)
        self.fingerprint_variable = attention_layer.fingerprint
        service_attention = attention_layer([service_contexts_embedding, service_fingerprints])

        x = tf.concat([observation_embedding, service_attention], axis=1)
        for spec in self.layer_specs:
            x = spec.instantiate()(x)
        satisfaction = tf.keras.layers.Dense(units=1, activation=None, name='satisfaction')(x)
        interference = tf.keras.layers.Dense(units=1, activation=None, name='interference')(x)

        model = tf.keras.Model(
            inputs=[observation, service_fingerprints, service_contexts],
            outputs=[satisfaction, interference],
        )
        return model

    def align_inputs(self, samples: List[Dict]):
        observation = [sample["observation"] for sample in samples]
        contexts = [sample["context"] for sample in samples]

        def get_fingerprints(context):
            return [
                context["fingerprints"][service] for service in context["services"]
                if service != self.service
            ] if context["services"] else np.zeros(shape=(1, self.embedding_size))

        def get_service_states(context):
            return [
                context["services"][service] for service in context["services"]
                if service != self.service
            ] if context["services"] else np.zeros(shape=(1, self.service_state_size))

        service_fingerprints = [get_fingerprints(context) for context in contexts]
        service_contexts = [get_service_states(context) for context in contexts]

        return {
            "observation": np.array(observation),
            "service_fingerprints": tf.ragged.constant(service_fingerprints, dtype=tf.float32, ragged_rank=1)
            if self.ragged else tf.constant(service_fingerprints, dtype=tf.float32),
            "service_contexts": tf.ragged.constant(service_contexts, dtype=tf.float32, ragged_rank=1)
            if self.ragged else tf.constant(service_contexts, dtype=tf.float32),
        }

    def align_outputs(self, samples: List[Dict]):
        return {
            "satisfaction": np.array([[sample["satisfaction"]] for sample in samples]),
            "interference": np.array([[sample["interference"]] for sample in samples]),
        }


class EqualAttentionAgent(ServiceAgent):
    def build_model(self) -> tf.keras.Model:
        # Model
        observation = tf.keras.Input(shape=(self.user_state_size,), dtype=tf.float32, name='observation')
        observation_embedding = tf.keras.layers.Dense(
            units=self.embedding_size, activation=self.embedding_activation,
        )(observation)

        service_contexts = tf.keras.Input(
            shape=(None, self.service_state_size), dtype=tf.float32, name='service_contexts',
            ragged=self.ragged,
        )
        service_contexts_embedding = tf.keras.layers.Dense(
            units=self.embedding_size, activation=self.embedding_activation,
            use_bias=False,
        )(service_contexts)

        service_attention = tf.math.reduce_mean(service_contexts_embedding, axis=-2)

        x = tf.concat([observation_embedding, service_attention], axis=1)
        for spec in self.layer_specs:
            x = spec.instantiate()(x)
        satisfaction = tf.keras.layers.Dense(units=1, activation=None, name='satisfaction')(x)
        interference = tf.keras.layers.Dense(units=1, activation=None, name='interference')(x)

        model = tf.keras.Model(
            inputs=[observation, service_contexts],
            outputs=[satisfaction, interference],
        )
        return model

    def align_inputs(self, samples: List[Dict]):
        observation = [sample["observation"] for sample in samples]
        contexts = [sample["context"] for sample in samples]

        def get_service_states(context):
            return [
                context["services"][service] for service in context["services"]
                if service != self.service
            ] if context["services"] else np.zeros(shape=(1, self.service_state_size))

        service_contexts = [get_service_states(context) for context in contexts]

        return {
            "observation": np.array(observation),
            "service_contexts": tf.ragged.constant(service_contexts, dtype=tf.float32, ragged_rank=1)
            if self.ragged else tf.constant(service_contexts, dtype=tf.float32),
        }

    def align_outputs(self, samples: List[Dict]):
        return {
            "satisfaction": np.array([[sample["satisfaction"]] for sample in samples]),
            "interference": np.array([[sample["interference"]] for sample in samples]),
        }
