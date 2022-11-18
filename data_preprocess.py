import pandas as pd
import string
import numpy as np
from keras.utils import to_categorical
import argparse 

def tokenizer(alphabet,url_length=200):
    dictionary_size = len(alphabet) + 1
    url_shape = (url_length, dictionary_size)
    dictionary = {}
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        dictionary[c]=i+1
        reverse_dictionary[i+1]=c
    return dictionary, reverse_dictionary
    
def data_npz(good,bad,alphabet,dictionary,samples=50000,url_length=200,npz_filename='phishing.npz'):
        #Xu Li URL GOOD
        good_data = []   
        i = 0
        for i in range(392157):
            line = good['URL'][i]
            this_sample=np.zeros(url_shape)   
            line = line.lower()
            for i, position in enumerate(this_sample):
                this_sample[i][0]=1.0
            for i, char in enumerate(line):
                this_sample[i][0]=0.0
                this_sample[i][dictionary[char]]=1.0
            good_data.append(this_sample)    
        good_data = np.array(good_data)
        # good_data = good_data[:samples]
        print ("Done Good")
        #Xu Li URL BAD
        bad_data = []   
        i = 0
        for i in range(149092):
            line = bad['URL'][i]
            this_sample=np.zeros(url_shape)
            line = line.lower()
            for i, position in enumerate(this_sample):
                this_sample[i][0]=1.0
            for i, char in enumerate(line):
                this_sample[i][0]=0.0
                this_sample[i][dictionary[char]]=1.0            
            bad_data.append(this_sample)
        bad_data = np.array(bad_data)
        print("--------------------------------------------------------")
        print ("Array Shape:", good_data.shape)
        print ("Array Shape:", bad_data.shape)
        x_train = np.concatenate((good_data[:312000,:,:], bad_data[:120000,:,:]),axis=0)
        x_test = np.concatenate((good_data[312000:392157,:,:], bad_data[120000:149092,:,:]),axis=0)
        good_label = np.ones((392157,1))
        bad_label = np.zeros((149092,1))
        y_train = np.concatenate((good_label[:312000,:], bad_label[:120000,:]),axis=0)        
        y_train_cat = to_categorical(y_train)
        y_test = np.concatenate((good_label[312000:392157,:], bad_label[120000:149092,:]),axis=0)
        y_test_cat = to_categorical(y_test)

        np.savez_compressed(npz_filename, X_train=x_train, X_test=x_test, y_train=y_train_cat, y_test=y_test_cat)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--url_length', type=int, default=200)
    parser.add_argument('--npz_filename', type=str, default='phishing.npz')
    parser.add_argument('--n_samples',type=int, default=140000,help='number of good and bad samples.')
    args = parser.parse_args()

    
    alphabet = string.ascii_lowercase + string.digits + "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
    dictionary_size = len(alphabet) + 1
    url_shape = (args.url_length, dictionary_size)
    df_good = pd.read_csv('.\\datasetcsv\\good.csv')
    df_bad = pd.read_csv('.\\datasetcsv\\bad.csv')
    good = df_good[df_good['Label']=='good']
    bad = df_bad[df_bad['Label']=='bad']
    good.reset_index(drop=True, inplace=True)
    bad.reset_index(drop=True, inplace=True)
    each_class_samples= args.n_samples #2
    dictionary, reverse_dictionary = tokenizer(alphabet,url_length= args.url_length)

  
    data_npz(good,bad,alphabet,dictionary,samples=each_class_samples,url_length=args.url_length,npz_filename=args.npz_filename)

    

