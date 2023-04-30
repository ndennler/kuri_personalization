from irlpreference import query_generation
import rospy
from std_msgs.msg import Int8, Int8MultiArray, String

import irlpreference as pref

class PreferenceEngine:

    def __init__(self):
        rospy.init_node('preference_engine_node')

        self.visual_embedings = None
        self.auditory_embeddings = None
        self.kinethetic_embeddings = None

        rospy.Subscriber('visual_choice', Int8, self.process_visual_choice)
        rospy.Subscriber('auditory_choice', Int8, self.process_auditory_choice)
        rospy.Subscriber('kinesthetic_choice', Int8, self.process_kinesthetic_choice)

        self.visual_query_pub = rospy.Publisher('visual_query', String, queue_size=1)
        self.auditory_query_pub = rospy.Publisher('auditory_query', String, queue_size=1)
        self.kinesthetic_query_pub = rospy.Publisher('kinesthetic_query', String, queue_size=1)

        print('ready')


    def process_visual_choice(self, msg):
        self.process_choice('visual', msg.data)

    def process_auditory_choice(self, msg):
        self.process_choice('auditory', msg.data)
    
    def process_kinesthetic_choice(self, msg):
        self.process_choice('kinesthetic', msg.data)

    def process_choice(self, modality, choice):
        print(f'{modality}: {choice}')
        if modality == 'visual':
            self.visual_query_pub.publish('7,8,9')
        elif modality == 'auditory':
            self.auditory_query_pub.publish('3,6,9')
        elif modality == 'kinesthetic':
            self.kinesthetic_query_pub.publish("4,5,6")
        


if __name__ == '__main__':
    peng = PreferenceEngine()
    rospy.spin()
