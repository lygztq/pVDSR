#hpyer = {
#	'DATA_PATH' : "./data/train/", 	# The training data path
#	'TEST_DATA_PATH':"./data/test/",# The test data path
#	'IMG_SIZE' : (41, 41), 			# The train data instance size
#	'BATCH_SIZE' : 64, 				# The Training batch size
#	'BASE_LR' : 0.0001,				# The initial learning step	
#	'LR_RATE' : 0.1, 				# The decay rate
#	'LR_STEP_SIZE' : 120,			# The epoch size
#	'MAX_EPOCH' : 120,
#	'USE_QUEUE_LOADING' : True		# Whether using queue loading
#
#}

DATA_PATH = "./data/train/"
IMG_SIZE = (41, 41)
BATCH_SIZE = 64
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 120
USE_QUEUE_LOADING = True
TEST_DATA_PATH = "./data/test/"

