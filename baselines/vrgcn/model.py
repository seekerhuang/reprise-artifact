def install(native_models, native_layers, tf, initial_supplier):
    original = native_models.GCN._build

    class Hidden(native_layers.Layer):

        def __init__(self, input_dim, hidden, dropout, cvd, **kw):
            super().__init__(**kw)
            self.input_dim = input_dim
            self.dropout = dropout
            self.cvd = cvd
            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, hidden), dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
                self.vars['bias'] = tf.get_variable('bias', shape=(hidden,), dtype=tf.float32, initializer=tf.zeros_initializer())

        def _call(self, inputs):
            inputs.set_shape([None, self.input_dim])
            h = tf.nn.relu(native_layers.dot(inputs, self.vars['weights']) + self.vars['bias'])
            if self.cvd:
                mu = tf.nn.relu(native_layers.dot(inputs, self.vars['weights']) + self.vars['bias'])
                dropped = tf.nn.dropout(h, keep_prob=1 - self.dropout)
                self.current_hidden = h
                self.dropped_hidden = dropped
                self.mean_hidden = mu
                return (dropped, tf.stop_gradient(mu))
            return tf.nn.dropout(h, keep_prob=1 - self.dropout)

    class Readout(native_layers.Layer):

        def __init__(self, hidden, classes, **kw):
            super().__init__(**kw)
            self.hidden = hidden
            with tf.variable_scope(self.name + '_vars'):
                self.vars['weights'] = tf.get_variable('weights', shape=(hidden, classes), dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
                self.vars['bias'] = tf.get_variable('bias', shape=(classes,), dtype=tf.float32, initializer=tf.zeros_initializer())

        def _call(self, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            x.set_shape([None, self.hidden])
            return native_layers.dot(x, self.vars['weights']) + self.vars['bias']

    def common_build(self):
        flags = native_models.FLAGS
        if flags.normalization != 'gcn' or flags.layer_norm or flags.num_fc_layers != 1 or (flags.num_layers != 2) or (not self.preprocess) or (self.L != 1) or flags.det_dropout or flags.reverse:
            raise RuntimeError('Shared GCN2/oneFC/noLN/noinputdrop/PP contract required')
        hidden = flags.hidden1
        first = Hidden(self.input_dim, hidden, self.placeholders['dropout'], self.cvd, name='dense0', logging=self.logging)
        last = Readout(hidden, self.output_dim, name='dense1', logging=self.logging)
        self.layers = [first, self.aggregators[0], last]
        self.layer_comp = [(self.input_dim * hidden, 0), (hidden * self.output_dim, 1)]
        self.canonical_variables = {}
        for (i, layer) in enumerate((first, last)):
            self.canonical_variables[f'convs.{i}.bias'] = (layer.vars['bias'], False)
            self.canonical_variables[f'convs.{i}.lin.weight'] = (layer.vars['weights'], True)
        for (key, value) in initial_supplier().items():
            if not key.startswith('bns.'):
                continue
            var = tf.get_variable(key.replace('.', '_'), initializer=value, dtype=tf.int64 if value.dtype.kind in 'iu' else tf.float32, trainable=key.endswith('.weight') or key.endswith('.bias'))
            self.canonical_variables[key] = (var, False)
    native_models.GCN._build = common_build
    return original
