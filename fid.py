from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.decomposition import PCA
from random import choices
from scipy.linalg import sqrtm

base_model = InceptionV3(weights='imagenet', include_top=False)

r_img_data = np.load('/content/drive/MyDrive/Colab Notebooks/E1 213 PRNN/r_img_data.npy')
r_inceptionV3_feature = base_model.predict(r_img_data)      
r_inceptionV3_feature = r_inceptionV3_feature.reshape(r_inceptionV3_feature.shape[0],-1)
r_img_data = r_img_data.reshape(r_img_data.shape[0],-1)
num_samples = r_img_data.shape[0]

def compute_fid(r_feature,s_feature):
    r_mu = np.mean(r_feature,axis=0)
    s_mu = np.mean(s_feature,axis=0)
    
    r_cov = np.cov(r_feature,rowvar=False)
    s_cov = np.cov(s_feature,rowvar=False)
    
    diff_mu = r_mu - s_mu
    
    f1 = np.linalg.norm(diff_mu)**2
    f2 = np.trace(r_cov + s_cov - 2*sqrtm(r_cov.dot(s_cov)).real)
    
    fid = f1 + f2
    
    return fid