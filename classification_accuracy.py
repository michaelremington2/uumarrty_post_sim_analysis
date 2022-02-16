#!/usr/bin/python
import pandas as pd
import numpy as np


class CalcPerCycleAccuracy(object):
    def __init__(self,actual_optima_points,test_set,x_range=0,y_range=0,padded=False):
        self.actual_optima_points = np.array(actual_optima_points, dtype = [("generation", int), 
                                                                           ("y", float), 
                                                                           ("class", np.unicode_,3)])
        self.test_set = test_set
        self.x_range=x_range
        self.y_range=y_range
        self.test_set=test_set
        self.true_positive_x = []
        self.true_positive_y = []
        self.false_positive_x = []
        self.false_positive_y = []
        self.true_negative_x = []
        self.true_negative_y = []
        self.false_negative_x = []
        self.false_negative_y = []
        self.true_positive_min_max=[]
        self.false_positive_min_max=[]
        self.true_negative_min_max=[]
        self.false_negative_min_max=[]
        self.false_positives=[]
        self.false_negatives=[]
        self.true_negatives=[]
        self.true_positives=[]
        if padded:
            self.padded_optima=self.add_acceptable_values()
        else:
            self.padded_optima=self.actual_optima_points

    def add_acceptable_values(self):
        padded_optima = []
        for optima in self.actual_optima_points:
            padded_optima.append(tuple(optima))
            for i in range(self.x_range):
                if int(optima[0])>=self.x_range:
                    back = int(optima[0])-i-1
                    back_point = (back, optima[1], optima[2])
                    padded_optima.append(back_point)
                    forward = int(optima[0])+i+1
                    forward_point = (forward, optima[1], optima[2])
                    padded_optima.append(forward_point)
        padded_optima = np.array(padded_optima, dtype = [("generation", int), 
                                                         ("y", float), 
                                                         ("class", np.unicode_,3)])
        return padded_optima
    
    def classify_true_and_false_positives(self):
        for index, row in self.test_set.iterrows():
            if row['point_classifacation'] in ['min','max'] and row['x'] in self.padded_optima['generation']:
                self.true_positive_x.append(row['x'])
                for element in self.padded_optima:
                    if int(element['generation']) == row['x']:
                        self.true_positives.append([row['x'],row['y'],row['point_classifacation']])
                        #check_y_value
                        if (float(element['y'])-self.y_range) <= row['y'] and (float(element['y'])+self.y_range) >= row['y']:
                            self.true_positive_y.append(row['y'])
                        else:
                            self.false_positive_y.append(row['y'])
                        #check_min_max
                        if str(element['class']) == row['point_classifacation']:
                            self.true_positive_min_max.append(row['point_classifacation'])
                        else:
                            self.false_positive_y.append(row['point_classifacation'])
            elif row['point_classifacation'] in ['min','max'] and row['x'] not in self.padded_optima['generation']:
                self.false_positive_x.append(row['x'])
                self.false_positive_y.append(row['y'])
                self.false_positive_min_max.append(row['point_classifacation'])
                self.false_positives.append([row['x'],row['y'],row['point_classifacation']])
            elif row['point_classifacation'] not in ['min','max'] and row['x'] not in self.padded_optima['generation']:
                self.true_negative_x.append(row['x'])
                self.true_negative_y.append(row['y'])
                self.true_negative_min_max.append(row['point_classifacation'])
                self.true_negatives.append([row['x'],row['y'],row['point_classifacation']])


    def classify_false_negatives(self):
        optima_test_set = self.test_set[self.test_set['point_classifacation'].isin(['min','max'])]
        for optima in self.actual_optima_points:
            if optima['generation'] not in list(optima_test_set['x']):
                self.false_negative_x.append(optima['generation'])
                self.false_negative_y.append(optima['y'])
                self.false_negative_min_max.append(optima['class'])
                self.false_negatives.append(optima)

    def calc_accuracy(self,true_positive_list,true_negative_list,test_set):
        return float(len(true_positive_list)+len(true_negative_list))/float(len(test_set))

    def calc_recall(self,true_positive_list,false_negaitve_list):
        return float(len(true_positive_list))/float((len(true_positive_list)+len(false_negaitve_list)))

    def calc_precision(self,true_positive_list,false_positive_list):
        return float(len(true_positive_list))/float((len(true_positive_list)+len(false_positive_list)))

    def print_confusion_matrix_total(self):
        print('True Positive: {}'.format(len(self.true_positives)))
        print('False Positive: {}'.format(len(self.false_positives)))
        print('False Negative: {}'.format(len(self.false_negatives)))
        print('True Negative: {}'.format(len(self.true_negatives)))

    def test_confusion_matrix(self):
        classified_points=len(self.true_positives)+len(self.false_positives)+len(self.false_negatives)+len(self.true_negatives)
        test_set_len=len(self.test_set)
        if classified_points==test_set_len:
            print('PASS')
        else:
            print('FAIL')
        print('Test Set Len:{}'.format(len(self.test_set)))
        print('Classified Set Len:{}'.format(classified_points))
        print('Actual Set Len:{}'.format(len(self.actual_optima_points)))
        print('Padded Set Len:{}'.format(len(self.padded_optima)))



    def main(self):
        self.classify_true_and_false_positives()
        self.classify_false_negatives()



if __name__ ==  "__main__":
    temp = CalcPerCycleAccuracy(actual_optima_points=snake_exp1_optima,
                              x_range = 1,
                              y_range = 0.01)
    temp.main()

