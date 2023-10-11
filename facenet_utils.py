import os
import time
import datetime
import json
import uuid
import numpy as np
import mysql.connector
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import joblib
# import config as tc  # Duplicate import
# import db_query_prod as db  # commented out import

# Database configs
DATABASE_CONFIG = {
    'host': "localhost",
    'user': "root",
    'port': "3306",
    'password': "Bharat1947",
    'database': "face_test"
}

# Building ID 
GATE_ID = {
    '0': ("cam1", "192.168.177.76"),
    '1': ("cam2", "192.168.1.12")
}

# Camera configs
VIDEO_PATH = "file:///home/asdael/Downloads/Vishwajeet/Metro-FRS-DS-Video/new3.mp4"
CAM1_PATH = "rtsp://ranu:Bharat1947@192.168.177.76:554/Profile2/media.smp"
CAM_PATHS = [VIDEO_PATH, CAM1_PATH]
print(CAM_PATHS)

NAMES_DICT = {}
COUNT = 0
KNN_CLASSIFIER_DICT = {"trained": False, "classifier": None}


def connect_to_db():
    return mysql.connector.connect(
        host=DATABASE_CONFIG['host'],
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        port=DATABASE_CONFIG['port'],
        database=DATABASE_CONFIG['database']
    )


def data_to_db(predicted_name, data_to_send, timestamp):
    db = connect_to_db()
    cursor = db.cursor()

    sql = "INSERT INTO Recog (PersonID, RecogPoint, TimeStamp) VALUES (%s, %s, %s)"
    val = (predicted_name, data_to_send, timestamp)

    cursor.execute(sql, val)
    db.commit()

    print(cursor.rowcount, "record inserted.")


def load_dataset(dataset_path):
    dataset_embeddings = np.load(dataset_path)
    faces_embeddings, labels = dataset_embeddings['arr_0'], dataset_embeddings['arr_1']
    faces_embeddings = faces_embeddings.reshape(-1, 128)
    faces_embeddings = normalize_vectors(faces_embeddings)
    return faces_embeddings, labels


def normalize_vectors(vectors):
    normalizer = Normalizer(norm='l2')
    return normalizer.transform(vectors)


def labels_encoder(labels):
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)
    return out_encoder, out_encoder.transform(labels)




def predict_using_classifier(faces_embeddings, labels, face_to_predict_embedding,DataToSend):
	x = datetime.datetime.now()
	TimeStamp = x.strftime("%x %X")
	uid = '0'
	classifier = get_classifier(faces_embeddings, labels, "knn")
	out_encoder, labels = labels_encoder(labels)
	# print(labels)
	face_to_predict_embedding = normalize_vectors(face_to_predict_embedding)
	# faces_embedding = normalize_vectors(faces_embeddings)
	face_to_predict_embedding = face_to_predict_embedding[0]
	# prediction for the face
	samples = np.expand_dims(face_to_predict_embedding, axis=0)
	# print(samples.shape)
	# If error raised check using dataset embeddings and classifier data are the same, remove classifier files
	yhat_class = classifier.predict(samples)
	# print(yhat_class)
	yhat_prob = classifier.predict_proba(samples)
	class_index = yhat_class[0]
	class_probability = (yhat_prob[0, class_index] )
	#print('class_index: ',class_index)
	# print('class_probability',class_probability)
	if class_probability > .53 and class_probability != 1.0:
		uid = str(uuid.uuid1())
		predicted_name = out_encoder.inverse_transform(yhat_class)[0]
		predicted_name = str(predicted_name)
		# print(type(predicted_name),type(DataToSend),type(TimeStamp))
		print(f" Initial Predicted = {predicted_name} with {class_probability} from source {DataToSend} at {TimeStamp}")
		#DataToDB(predicted_name,DataToSend,TimeStamp)
	else:
		predicted_name = 'Unknown'
		print(predicted_name,class_probability)
	# predicted_name += f' {class_probability}'
	# print("predicted_name ", predicted_name, " class_probability ", class_probability)
	return predicted_name, class_probability,uid

def get_classifier(faces_embeddings=None, labels=None, classifier_name='knn'):
	'''
			The method will load the classifier if it exist in the specified path
			otherwise, it will train the classifier using faces_embeddings and labels
	'''

	if (KNN_CLASSIFIER_DICT["trained"] == False):
		#print('[INFO] Training Classifier...')
		classifier = train_classifier(faces_embeddings, labels, classifier_name=classifier_name)
		KNN_CLASSIFIER_DICT["classifier"] = classifier
		KNN_CLASSIFIER_DICT["trained"] = True
		# print(CLASSIFIER_DICT[classifier_name]["trained"])
	else:
		#print('[INFO] Trained classifier assigned...')
		classifier = KNN_CLASSIFIER_DICT["classifier"]
	#print('[INFO] Classifier Training Done...')
	return classifier

def train_classifier(faces_embeddings, labels, classifier_name='knn'):
	'''
			The method will train the classifier. There are three classifiers:
			- KNN
			- SVM
			- Neural Network
	'''

	# print('---------------',faces_embeddings.shape)
	faces_embeddings = normalize_vectors(faces_embeddings)
	out_encoder, labels = labels_encoder(labels)
	
	if classifier_name == 'knn':
		classifier = KNeighborsClassifier(n_neighbors=2, p=1, weights="distance", metric="euclidean")
	else:
		raise ValueError('Classifier name not found, classifier should be: knn, svm, neural_network')

	# fit model
	classifier.fit(faces_embeddings, labels)

	# save the classifier to file
	save_path = os.path.join(os.getcwd(), 'classifiers',
							'{}.sav'.format(classifier_name))
	joblib.dump(classifier, save_path)
	print('Classifier Saved to [%s]...' % save_path)
	return classifier


