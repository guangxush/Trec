#coding=utf-8
'''
计算出行方向
'''
def direction_index(s_x, s_y, e_x, e_y):
        '''                       
        0: still                  
        1: forward                
        2: back                   
        3: left front             
        4: left back              
        5: left                   
        6: right front            
        7: right back             
        8: right
        '''
        if s_x == e_x:            
                if s_y == e_y:    
                        return 0  
                elif s_y < e_y:   
                        return 1  
                elif s_y > e_y:   
                        return 2  
        elif s_x > e_x:           
                if s_y < e_y:     
                        return 3  
                elif s_y > e_y:   
                        return 4  
                else:             
                        return 5  
        else:                     
                if s_y < e_y:     
                        return 6  
                elif s_y > e_y:   
                        return 7  
                else:             
                        return 8 

