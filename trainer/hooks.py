import tensorflow as tf

class PGTrainHook(tf.train.SessionRunHook):
    """Some documentation"""

    def __init__(self, 
            G_train_op, 
            D_train_op, 
            ALPHA, 
            RES,
            stablize_increment,
            fade_increment,
            res_increment,
            reset_alpha):
        super(PGTrainHook, self).__init__()

        self.G_train_op = G_train_op,
        self.D_train_op = D_train_op,
        self.ALPHA = ALPHA
        self.RES = RES
        self.stablize_increment = stablize_increment
        self.fade_increment = fade_increment
        self.res_increment = res_increment
        self.reset_alpha = reset_alpha


    def before_run(self, run_context):
        alpha = run_context.session.run(self.ALPHA)
        if alpha < 0:      # Stablize layer
            run_context.session.run(self.stablize_increment)
        elif alpha < 1:    # Fade in new layer
            alpha = run_context.session.run(self.fade_increment)
        else:                   # Increase Resolution
            res, alpha = run_context.session.run([self.res_increment, self.reset_alpha])
            print('Increasing resolution to {}\nalpha reset to {}'.format(res, alpha))

        run_context.session.run(self.G_train_op)
        run_context.session.run(self.D_train_op)
