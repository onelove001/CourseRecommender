#Importing the required libraries

import pandas as pd 
import numpy as np
import random
from playsound import playsound
import speech_recognition as sr
import re 
import gtts
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

#importing the dataset
coursera_dataset=pd.read_csv('Dataset/ml-latest/courses.csv')
ratings_dataset=pd.read_csv('Dataset/ml-latest/ratings.csv')

#Converting the format of Level column to a list and then appending to the new list
Level=[]
Levels={}

for num in range(0,len(coursera_dataset)):
    key=coursera_dataset.iloc[num]['title']
    value=str(coursera_dataset.iloc[num]['level']).split("|")
    Levels[key]=value
    Level.append(value)

#Making a new column in our original Dataset         
coursera_dataset['new'] = Level

courses_name=[]
raw=[]

for courses in coursera_dataset['title']:
      courses_name.append(courses)

coursera_dataset['course_name']=courses_name

#Converting the datatype of new column from list to string as required by the function
coursera_dataset['new']=coursera_dataset['new'].apply(' '.join)

'''Applying the Cotent Based Filtering'''
 #Applying Feature extraction 
from sklearn.feature_extraction.text import TfidfVectorizer

tfid=TfidfVectorizer(stop_words='english')
#matrix after applying the tfidf
matrix=tfid.fit_transform(coursera_dataset['new'])

#Compute the cosine similarity of every level
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim=cosine_similarity(matrix, matrix)

'''Applying the Collaborative Filtering'''
#Intialising the Reader which is used to parse the file containing the ratings 
reader=Reader()

#Making the dataset containing the column as userid itemid ratings
dataset=Dataset.load_from_df(ratings_dataset[['userId','course_id','rating']],reader)

#Intialising the SVD model and specifying the number of latent features
svd=SVD(n_factors=25)

#performing cross validation 
#evaluting the model on the based on the root mean square error and Mean absolute error 
cross_validate(svd, dataset, measures=['rmse','mae'], cv=6)

#making the dataset to train our model
train=dataset.build_full_trainset()

#training our model
svd.fit(train)

#Making a new series which have two columns in it 
#Course name and course id 
coursera_dataset = coursera_dataset.reset_index()
titles = coursera_dataset['course_name']
indices = pd.Series(coursera_dataset.index, index=coursera_dataset['course_name'])


#Function to make recommendation to the user
def recommendataion(user_id, course):
    result=[]
    #Getting the id of the course for which the user want recommendation
    ind=indices[course]
    #Getting all the similar cosine score for that course
    sim_scores=list(enumerate(cosine_sim[ind]))
    #Sorting the list obtained
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)    
    #Getting all the id of the courses that are related to the course entered by the user
    course_id=[i[0] for i in sim_scores]
    print(" ")
    print('The Courses You Should Study Next Are -: ')
    print(" ")
    print("------------------------------------------------------------------------------------------------------------")
    print(" ")
    print('CourseID  | Course Name  |  Average Rating  |  Predicted Rating  --> ')

    #Varible to print only top 10 courses
    count=0
    for id in range(0,len(course_id)):
      #to ensure that the course entered by the user is does not come in his/her recommendation
        if(ind != course_id[id]):
            ratings=ratings_dataset[ratings_dataset['course_id']==course_id[id]]['rating']
            avg_ratings=round(np.mean(ratings),2)
            #For getting all the courses that a particular user has rated
            rated_courses=list(ratings_dataset[ratings_dataset['userId']==user_id]['course_id'])
            #to take only thoese courses that a particular user and not gotten yet
            if(id not in rated_courses):
                #To print only thoese movies which have an average ratings that is more than 3.5
                if(avg_ratings > 3.5):
                    count+=1
                    #Getting the course_id of the corresponding course_name
                    id_courses=coursera_dataset[coursera_dataset['course_name']==titles[course_id[id]]]['course_id'].iloc[0]
                    predicted_ratings=round(svd.predict(user_id,course_id[id]).est,2)
                    print(f'{course_id[id]} , {titles[course_id[id]]} ,{avg_ratings}, {predicted_ratings}')
                    result.append([titles[course_id[id]],str('Predicted Rating'),str(predicted_ratings)])
                if(count >= 10):
                        break
    return result


#Converting the speech to text using google text to speech api
def speech_to_text(): 
    text=''
    sample_rate = 48000
    chunk_size = 2048

    #Initialize the recognizer 
    r = sr.Recognizer() 
    with sr.Microphone(sample_rate = sample_rate, chunk_size = chunk_size) as source:

        #waiting to let the recognizer adjust the energy threshold based on the surrounding noise level 
        r.adjust_for_ambient_noise(source)
        print(" ") 
        print ("Speak the name of the course ")
        #listens for the user's input 
        audio = r.listen(source) 
        try: 
            text = r.recognize_google(audio) 
            print(text.title())
        #error occurs when google could not understand what was said 
        except sr.UnknownValueError:
            print(" ")
            print("Google Speech Recognition could not understand audio") 
          
        except sr.RequestError as e:
            print(" ")
            print("Could not request results from Google  Speech Recognition service; {0}".format(e)) 
    return text.title()        


#Converting the text to speech using google text to speech api
def text_to_speech():
    print(" ")
    username = input("Enter Your Name: ")
    file = open('dd.txt','w')
    file.writelines(f'hello {username} \n')
    file.writelines('The courses you should study  next and their predicted ratings are as follows:  \n') 
    for res in result:
        res=' '.join(res)
        file.write(res+'\n')
        
    file.close()
    file = open('dd.txt','r')
    data= file.read()
    file.close()
    
    language = 'en'
      
    # Passing the text and language to the engine,  
    myobj = gtts.gTTS(text=data, lang=language, slow=False) 
      
    # Saving the converted audio in a mp3 file named   
    i=random.randint(1,100)
    file='new'+str(i)+'.mp3'
    myobj.save(file)      
    playsound(file)
        
#Getting the output   
try:
    print(" ")
    i_d=int(input('Enter Your UserID: '))
    print(" ")
    #Uncomment the next code line if you wish to enter the course by voice print
    #course_name=speech_to_text()
    #Comment the next code line to use a voice print
    course_name = input("Enter course title: ")
    result=recommendataion(i_d, course_name)
    text_to_speech()
except:
    print(" ")
    print("Sorry!❌. The course entered could not be located!❌")