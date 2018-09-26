
#Logistic Regression vectorized:

with tf.name_scope('training_data') as scope:
    X = tf.placeholder(tf.float64, shape=(None,X_train.shape[1]+1), name='X')
    y = tf.placeholder(tf.float64, shape=(None,1), name='y')
    X_extended = np.c_[np.ones(X_train.shape[0]),X_train]
    y_t = tf.transpose(y)

theta = tf.Variable(np.zeros([X_train.shape[1]+1,1]), name='theta')

#p = 1/(1+exp(-theta*x))
with tf.name_scope('h_x') as scope:
    p = 1/(1+tf.exp(-tf.matmul(X, theta)))
#cost = (1/m)*(-y.T * log(p) - (1-y.T)*log(1-p))
with tf.name_scope('cost') as scope:
    # or (1/batch_size) ?
    cost = (1/X_train.shape[0])*(-tf.matmul(y_t, tf.log(p)) - tf.matmul(1-y_t, tf.log(1-p)))