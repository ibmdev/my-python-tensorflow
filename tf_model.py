# Un programme TensorFlow = Construire un graphe de Tensors
# Propriétés d'un Tensor = 1 data type (float32, int32, or string) + 1 shape
# Types de Tensor  = tf.Variable, tf.constant, tf.placeholder, tf.SparseTensor
import tensorflow as tf
print('TensorFlow : ' + tf.__version__)

# Etape 1 : Préparation du modèle
# Un graphe TensorFlow qui implémente un réseau de neurones simple couche avec activations RELU
# Définition de 4 Tensors
# TI = Tensor entrée à 3 dimensions
# TW = Tensor de poids 3x2
# TB = Tensor de biais
# TO = Tensor d'activation
TI = tf.placeholder(tf.float32, shape=[None, 3], name='I')
TW = tf.Variable(tf.zeros(shape=[3, 2]), dtype=tf.float32, name='W')
Tb = tf.Variable(tf.zeros(shape=[2]), dtype=tf.float32, name='b')
TO = tf.nn.relu(tf.matmul(TI, TW) + Tb, name='O')
print('Préparation du modèle')
# Etape 2 : Sauvegarde du graphe de calcul dans un fichier sérialisé GraphDef (extension textuelle .pbtxt)
saver = tf.train.Saver()
init_operation = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_operation)
    tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')
print('Sauvegarde du graphe de calcul ')
