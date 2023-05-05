from urllib import request
from irlpreference import query_generation
import rospy
from std_msgs.msg import Int32, Int32MultiArray, String

import irlpreference as pref
from irlpreference.input_models import LuceShepardChoice
from irlpreference.reward_parameterizations import MonteCarloLinearReward

import pandas as pd
import numpy as np
import os

class PreferenceEngine:

    def __init__(self, PID):
        try:
            os.mkdir(f'../data/{PID}')
        except Exception as e:
            print(e)

        rospy.init_node('preference_engine_node')
        SIGNALS = ['idle', 'has_item', 'found_item', 'searching']
        np.random.shuffle(SIGNALS)
        self.signal_iter = iter(SIGNALS)

        '''
        get the stimulus info
        '''
        self.signal_name = next(self.signal_iter)
        print(f'Design for {self.signal_name}')
        self.PID = PID

        self.info = pd.read_csv('/home/icaros/Desktop/kuri_personalization/web_interfaces/data/all_data.csv')

        self.visual_embeddings = np.load('../../stimuli/visual/data/32_embeddings.npy')
        self.auditory_embeddings = np.load('../../stimuli/auditory/sound-effect-library/data/32_embeddings.npy')
        self.kinesthetic_embeddings = np.load('../../stimuli/kinesthetic/data/32_embeddings.npy')

        '''
        Set up the preference learning things
        '''
        print(self.visual_embeddings.shape,self.auditory_embeddings.shape,self.kinesthetic_embeddings.shape)

        self.choice_model = LuceShepardChoice()

        self.visual_pref = MonteCarloLinearReward(32, number_samples=10_000)
        self.auditory_pref = MonteCarloLinearReward(32, number_samples=10_000)
        self.kinesthetic_pref = MonteCarloLinearReward(32, number_samples=10_000)

        self.visual_query = self.info.query(f"type=='Video'")['id'].sample(n=3).values
        self.auditory_query = self.info.query(f"type=='Audio'")['id'].sample(n=3).values
        self.kinesthetic_query = self.info.query(f"type=='Movement'")['id'].sample(n=3).values
        print(self.kinesthetic_query)
        
        '''
        Set up ROS subscribers / publishers
        '''
        rospy.Subscriber('visual_choice', Int32, self.process_visual_choice)
        rospy.Subscriber('auditory_choice', Int32, self.process_auditory_choice)
        rospy.Subscriber('kinesthetic_choice', Int32, self.process_kinesthetic_choice)

        rospy.Subscriber('signal_done', String, self.move_to_next_signal)
        rospy.Subscriber('request_stimuli', String, self.handle_request)

        self.visual_query_pub = rospy.Publisher('visual_query', String, queue_size=1)
        self.auditory_query_pub = rospy.Publisher('auditory_query', String, queue_size=1)
        self.kinesthetic_query_pub = rospy.Publisher('kinesthetic_query', String, queue_size=1)

        self.stimuli_pub = rospy.Publisher('best_stimuli', String, queue_size=1)

        print(','.join([str(i) for i in self.visual_query]))

        rospy.sleep(0.5)
        self.visual_query_pub.publish(','.join([str(i) for i in self.visual_query])) #gotta clip off the brackets I guess
        
        rospy.sleep(.1)
        self.auditory_query_pub.publish(','.join([str(i) for i in self.auditory_query]))
        rospy.sleep(.1)
        self.kinesthetic_query_pub.publish(','.join([str(i) for i in self.kinesthetic_query]))

        # self.stimuli_pub.publish('0,1,2,3')
        print('ready')


    def process_visual_choice(self, msg):
        #get choice, tell the user estimate model
        choice = msg.data
        query = self.visual_embeddings[self.visual_query + [0]]
        query[-1, :] = -np.average(query[:3], axis=0)

        self.choice_model.tell_input(choice, query)
        self.visual_pref.update(self.choice_model.get_probability_of_input)

        #get new query, and send
        has_best = False
        for index in range(len(self.visual_query)):
            if index == choice:
                self.visual_query[index] = self.visual_query[index]
            elif not has_best:
                has_best = True
                best_indices = np.argpartition(self.choice_model.get_choice_probabilities(self.visual_embeddings,self.visual_pref.get_expectation()), -4)[-4:]
                for best_index in best_indices:
                    if best_index in self.info.query(f"type=='Video'")['id'].values \
                    and (choice > 2 or best_index != self.visual_query[choice]): #if the best isn't the same as the one that we are keeping, change it
                        self.visual_query[index] = best_index 
                        break
            else:
                self.visual_query[index] = self.info.query(f"type=='Video'")['id'].sample().values[0]
                # print()
        
        self.visual_query_pub.publish(str(self.visual_query)[1:-1])


    def process_auditory_choice(self, msg):
        #get choice, tell the user estimate model
        choice = msg.data
        query = self.auditory_embeddings[self.auditory_query + [0]]
        query[-1, :] = -np.average(query[:3], axis=0)

        self.choice_model.tell_input(choice, query)
        self.auditory_pref.update(self.choice_model.get_probability_of_input)

        #get new query, and send
        has_best = False
        for index in range(len(self.auditory_query)):
            if index == choice:
                self.auditory_query[index] = self.auditory_query[index]

            elif not has_best:
                has_best = True
                
                best_indices = np.argpartition(self.choice_model.get_choice_probabilities(self.auditory_embeddings,self.auditory_pref.get_expectation()), -4)[-4:]
                for best_index in best_indices:
                    if best_index in self.info.query(f"type=='Audio'")['id'].values \
                    and (choice > 2 or best_index != self.auditory_query[choice]): #if the best isn't the same as the one that we are keeping, change it
                        self.auditory_query[index] = best_index 
                        break
            else:
                self.auditory_query[index] = self.info.query(f"type=='Audio'")['id'].sample().values[0]
        
        self.auditory_query_pub.publish(str(self.auditory_query)[1:-1])
    
    def process_kinesthetic_choice(self, msg):
        #get choice, tell the user estimate model
        choice = msg.data
        query = self.kinesthetic_embeddings[self.kinesthetic_query + [0]]
        query[-1, :] = -np.average(query[:3], axis=0)

        self.choice_model.tell_input(choice, query)
        self.kinesthetic_pref.update(self.choice_model.get_probability_of_input)

        #get new query, and send
        has_best = False
        for index in range(len(self.kinesthetic_query)):
            if index == choice:
                self.kinesthetic_query[index] = self.kinesthetic_query[index]
            elif not has_best:
                has_best = True
                best_indices = np.argpartition(self.choice_model.get_choice_probabilities(self.kinesthetic_embeddings,self.kinesthetic_pref.get_expectation()), -4)[-4:]
                for best_index in best_indices:
                    if best_index in self.info.query(f"type=='Movement'")['id'].values \
                    and (choice > 2 or best_index != self.kinesthetic_query[choice]): #if the best isn't the same as the one that we are keeping, change it
                        self.kinesthetic_query[index] = best_index 
                        break
            else:
                self.kinesthetic_query[index] = self.info.query(f"type=='Movement'")['id'].sample().values[0]
        
        self.kinesthetic_query_pub.publish(str(self.kinesthetic_query)[1:-1])

    def move_to_next_signal(self, msg):

        np.savez_compressed(f'../data/{self.PID}/{self.signal_name}_end_results', 
            vis_h=self.visual_pref.hypothesis_samples,
            vis_p=self.visual_pref.hypothesis_log_probabilities, 
            aud_h = self.auditory_pref.hypothesis_samples, 
            aud_p = self.auditory_pref.hypothesis_log_probabilities,
            kin_h = self.kinesthetic_pref.hypothesis_samples,
            kin_p = self.kinesthetic_pref.hypothesis_log_probabilities)
        
        self.visual_pref.reset()
        self.auditory_pref.reset()
        self.kinesthetic_pref.reset()

        users_choice = msg.data
        print(users_choice)

        try:
            self.signal_name = next(self.signal_iter)
        except Exception as e:
            print('Done with personalizing signals.')

        print(f'Design for {self.signal_name}')
        self.visual_query = [np.random.randint(low=0, high=len(self.visual_embeddings)) for _ in range(3)]
        self.auditory_query = [np.random.randint(low=0, high=len(self.auditory_embeddings)) for _ in range(3)]
        self.kinesthetic_query = [np.random.randint(low=0, high=len(self.kinesthetic_embeddings)) for _ in range(3)]
        rospy.sleep(.1)
        self.visual_query_pub.publish(str(self.visual_query)[1:-1]) #gotta clip off the brackets I guess
        self.auditory_query_pub.publish(str(self.auditory_query)[1:-1])
        self.kinesthetic_query_pub.publish(str(self.kinesthetic_query)[1:-1])

    
    def handle_request(self, msg):
        NUM_RESPONSES = 100
        request_type, filters = msg.data.split(':')
        filters  = filters.split(',')

        candidates = self.info.query(f'type == "{request_type}"')

        #filter the tags
        if filters[0] != ' ':
            for filter in filters:
                candidates = candidates[candidates['tags'].str.contains(filter.strip())]

        #TODO: further filtering if there are atually filters
        if request_type == 'Video':
            embeds = self.visual_embeddings
            pref = self.visual_pref.get_expectation()
        if request_type == 'Audio':
            embeds = self.auditory_embeddings
            pref = self.auditory_pref.get_expectation()
        if request_type == 'Movement':
            embeds = self.kinesthetic_embeddings
            pref = self.kinesthetic_pref.get_expectation()

        indices = candidates['id'].to_numpy()
        embeds = embeds[indices]

        if len(embeds) > NUM_RESPONSES:
            best_indices = np.argpartition(self.choice_model.get_choice_probabilities(embeds,pref), -NUM_RESPONSES)[-NUM_RESPONSES:]
        else:
            best_indices = np.arange(len(indices))

        to_send = ','.join(map(str, indices[best_indices]))
        self.stimuli_pub.publish(to_send)
        print(len(best_indices))






        


if __name__ == '__main__':
    peng = PreferenceEngine(PID=1)
    rospy.spin()
