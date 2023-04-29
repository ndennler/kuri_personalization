import rospy


class PreferenceEngine:

    def __init__(self):
        self.visual_embedings = None
        self.auditory_embeddings = None
        self.kinethetic_embeddings = None



if __name__ == '__main__':
    peng = PreferenceEngine()
    rospy.spin()
