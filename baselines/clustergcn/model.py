def model_class(native_models, native_layers, tf, initial_values):

    class CommonConv(native_layers.Layer):

        def __init__(self, input_dim, output_dim, placeholders, *, precalc, dropout, act, canonical_index, **kwargs):
            super().__init__(**kwargs)
            self.precalc = precalc
            self.input_dim = input_dim
            self.act = act
            self.support = placeholders['support']
            self.dropout = placeholders['dropout'] if dropout else 0.0
            self.canonical_index = canonical_index
            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = tf.get_variable('weights', shape=[input_dim, output_dim], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
                self.vars['bias'] = tf.get_variable('bias', shape=[output_dim], dtype=tf.float32, initializer=tf.zeros_initializer())

        def _call(self, inputs):
            inputs.set_shape([None, self.input_dim])
            if self.precalc:
                message = inputs
            else:
                hidden = tf.nn.dropout(inputs, keep_prob=1 - self.dropout)
                message = tf.sparse_tensor_dense_matmul(self.support, hidden)
            output = tf.matmul(message, self.vars['weights']) + self.vars['bias']
            return self.act(output)

    class CommonGCN(native_models.Model):

        def __init__(self, placeholders, input_dim, **kwargs):
            super().__init__(**kwargs)
            if self.num_layers != 2 or not self.precalc or self.norm:
                raise ValueError('Common GCN requires2layers/precalc/noLayerNorm')
            self.inputs = placeholders['features']
            self.input_dim = input_dim
            self.output_dim = placeholders['labels'].shape.as_list()[1]
            self.placeholders = placeholders
            self.optimizer = tf.train.AdamOptimizer(learning_rate=native_models.FLAGS.learning_rate)
            self.build()

        def _build(self):
            hidden = native_models.FLAGS.hidden1
            self.layers = [CommonConv(self.input_dim, hidden, self.placeholders, precalc=True, dropout=False, act=tf.nn.relu, canonical_index=0, name='convs0', logging=self.logging), CommonConv(hidden, self.output_dim, self.placeholders, precalc=False, dropout=True, act=lambda x: x, canonical_index=1, name='convs1', logging=self.logging)]
            self.canonical_variables = {}
            for (i, layer) in enumerate(self.layers):
                self.canonical_variables['convs.' + str(i) + '.bias'] = (layer.vars['bias'], False)
                self.canonical_variables['convs.' + str(i) + '.lin.weight'] = (layer.vars['weights'], True)
            for (key, value) in initial_values.items():
                if not key.startswith('bns.'):
                    continue
                variable = tf.get_variable(key.replace('.', '_'), initializer=value, dtype=tf.int64 if value.dtype.kind in 'iu' else tf.float32, trainable=key.endswith('.weight') or key.endswith('.bias'))
                self.canonical_variables[key] = (variable, False)
    return CommonGCN
