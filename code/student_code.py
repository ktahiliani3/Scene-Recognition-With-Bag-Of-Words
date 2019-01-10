import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from IPython.core.debugger import set_trace
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix







def cross_validation(train_image_feats, train_labels, test_image_feats):


  folds = KFold(n_splits = 15, shuffle = True)
  folds.get_n_splits(train_image_feats)
  Accuracy = []
  SDeviation = []

  for train_index, test_index in folds.split(train_image_feats):

    NewTrain, NewLabelTrain = train_image_feats[train_index], train_image_feats[test_index]

    NewTest = []
    NewLabelTest = []

    for i in range(len(train_index)):

      NewTest.append(train_labels[train_index[i]])

    for j in range(len(test_index)):

      NewLabelTest.append(train_labels[test_index[j]])


    PredictedLabels = svm_classify(NewTrain, NewTest, NewLabelTrain)

    ConfMatrix = confusion_matrix(NewLabelTest, PredictedLabels)
    ConfMatrix = ConfMatrix.astype(np.float)/ConfMatrix.sum(axis = 1)[:, np.newaxis]
    Accuracy.append(np.mean(np.diag(ConfMatrix)))
    SDeviation = np.std(Accuracy)


  FinalAccuracy = (np.sum(Accuracy)*100/15)

  print(SDeviation)
  print(Accuracy)

  return FinalAccuracy





def get_tiny_images(image_paths):
  """
  This feature is inspired by the simple tiny images used as features in
  80 million tiny images: a large dataset for non-parametric object and
  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/


  Args:
  -   image_paths: list of N elements containing image paths

  Returns:
  -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
  """


  TotalImages = len(image_paths)
  Resize = 16
  feats = np.zeros((TotalImages, Resize*Resize))

  for i in range(TotalImages):

    #taking out each individual image from the given image path and resizing.

    Image = load_image_gray(image_paths[i])
    ResizedImage = cv2.resize(Image,(Resize,Resize))

    # creating a feature from the resized image;

    Feature = np.reshape(ResizedImage,(1,256))


    # zero mean and unit length



    FeatureNew = (Feature - np.mean(Feature))/np.std(Feature)

    #print(np.linalg.norm(FeatureNew, ord = 1))



    feats[i,:] = FeatureNew





  return feats

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.



  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """


  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))



  N = 400
  StepSize = 10

  TotalImages = len(image_paths)

  for i in range(TotalImages):

    Image = load_image_gray(image_paths[i])

    Frames, Descriptors = vlfeat.sift.dsift(Image, fast = 1, step = StepSize)

    Descriptors = np.random.randint(0, high = Descriptors.shape[0] - 1, size = (400,128))


    if i == 0:

      SIFT = np.stack(Descriptors)

    else:

      SIFT = np.vstack((SIFT,Descriptors))

  SIFT = SIFT.astype(float)






  ClusterCenters = vlfeat.kmeans.kmeans(SIFT, vocab_size)

  vocab = ClusterCenters



  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
  """


  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)




  vocab_size = 200
  TotalImages = len(image_paths)
  StepSize = 3
  feats = np.zeros((TotalImages, vocab_size))



  for i in range(TotalImages):

    Image = load_image_gray(image_paths[i])

    Frames, Descriptors = vlfeat.sift.dsift(Image, fast = 1, step = StepSize)

    Descriptors = Descriptors.astype(float)

    assignments = vlfeat.kmeans.kmeans_quantize(Descriptors, vocab)

    AssignmentHist, edges = np.histogram(assignments, bins = vocab_size, density = True)

    AssignmentHist = np.asarray(AssignmentHist)


    feats[i, : ] = AssignmentHist




  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
  """
  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """


  D = sklearn_pairwise.pairwise_distances(test_image_feats,train_image_feats)


  train_labels = np.asarray(train_labels)

  test_labels = []

  result = np.argsort(D, axis = 1)





  for i in range(len(result)):


    temp = result[i][0]
    test_labels = np.append(test_labels,train_labels[temp])






  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats):
  """


  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """



  # categories
  categories = list(set(train_labels))
  categories_copy = categories
  vocab_size = 200
  test = []

  # construct 1 vs all SVMs for each category
  svms = {cat: LinearSVC(random_state=0, tol=0.0000007, loss='hinge', C=1000, max_iter = 1000000000) for cat in categories}
  test_labels = []
  w = np.zeros((len(categories),vocab_size))
  b= np.zeros((len(categories),))


  for j in range(len(categories)):

    labels_temp = np.zeros((len(train_labels),1))

    for i in range(len(train_labels)):


      if categories[j] == train_labels[i]:

        labels_temp[i] = 1


      else:

        labels_temp[i] = 0

    labels_temp = np.asarray(labels_temp)
    labels_temp = np.ravel(labels_temp)


    model = svms[categories_copy[j]]
    model.fit(train_image_feats, labels_temp)
    w[j, :] = model.coef_
    b[j] = model.intercept_


  for k in range(test_image_feats.shape[0]):

    FeatureTemp = test_image_feats[k,:]

    ConfTemp = np.zeros((len(categories),))

    for l in range(len(categories)):

      tempW = w[l,:]
      ConfTemp[l] = np.dot(tempW, FeatureTemp) + b[l]

    maxind = np.argmax(ConfTemp)
    maxval = np.amax(ConfTemp)
    test.append(categories[maxind])

  test_labels = test
  return test_labels




def svm_nonlinear(train_image_feats, train_labels, test_image_feats):


  # categories
  categories = list(set(train_labels))
  categories_copy = categories
  vocab_size = 200
  test = []

  # construct 1 vs all SVMs for each category
  svms = {cat: SVC(C=3,kernel='rbf', degree=3, gamma= 190, coef0=0.0, shrinking=True, probability=False, tol=0.00000001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None) for cat in categories}
  test_labels = []
  w = np.zeros((len(categories), test_image_feats.shape[0]))
  b= np.zeros((len(categories),))


  for j in range(len(categories)):

    labels_temp = np.zeros((len(train_labels),1))

    for i in range(len(train_labels)):


      if categories[j] == train_labels[i]:

        labels_temp[i] = 1


      else:

        labels_temp[i] = 0

    labels_temp = np.asarray(labels_temp)
    labels_temp = np.ravel(labels_temp)


    model = svms[categories_copy[j]]
    model.fit(train_image_feats, labels_temp)
    w[j,:] = model.decision_function(test_image_feats)


  w = w.T

  for j in range(test_image_feats.shape[0]):

    maxind = np.argmax(w[j,:])
    test.append(categories[maxind])


  test_labels = test






  return test_labels
