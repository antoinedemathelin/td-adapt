# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 02:47:20 2023

@author: Boubo
"""
import numpy as np
import adapt._tree_utils as ut

def count_leaves_dt(dt,cl):
    count=0
    leaves = np.where(dt.tree_.feature == -2)[0]
    for l in leaves:
        if np.argmax(dt.tree_.value[l])==cl:
            count+=1
    return count
        
def count_leaves(rf,cl=1):
    counts = np.zeros(rf.n_estimators)
    for k,dt in enumerate(rf.estimators_):
        counts[k] = count_leaves_dt(dt,cl)
        
    return np.mean(counts)

def is_same_node(tree1,tree2,node1,node2,soft=False):

    if (node1 == -1 or node2 == -1):
        return False
    if tree1.tree_.feature[node1] != tree2.tree_.feature[node2]:
        return False
    if not soft:
        if tree1.tree_.threshold[node1] != tree2.tree_.threshold[node2]:
            return False
    
    return True

def highlight_different_nodes(tree1,tree2,node1,node2,soft=False):
    
    list_nodes1 = []
    list_nodes2 = []
    
    if is_same_node(tree1,tree2,node1,node2,soft=soft) :
        left1 = tree1.tree_.children_left[node1]
        right1 = tree1.tree_.children_right[node1]
        left2 = tree2.tree_.children_left[node2]
        right2 = tree2.tree_.children_right[node2]
        nl1, nl2 = highlight_different_nodes(tree1,tree2,left1,left2,soft=soft)
        nr1, nr2 = highlight_different_nodes(tree1,tree2,right1,right2,soft=soft)
        
        list_nodes1 = list_nodes1 + nl1 + nr1
        list_nodes2 = list_nodes2 + nl2 + nr2
    else:        
        list_nodes1 = list_nodes1 + ut.sub_nodes(tree1.tree_, node1)
        list_nodes2 = list_nodes2 + ut.sub_nodes(tree2.tree_, node2)
        
    return list_nodes1, list_nodes2
    
class TreeDot():
    def __init__(self,name_file,path=''):
        self.nodes = list()
        self.path = path
        self.path_file = path+name_file
        f = open(self.path_file,"r")
        self.lines = f.readlines()
        for line in self.lines :
            start = line.split(' ')[0]
            
            if start.isnumeric():
                self.nodes.append(int(start))
                
        self.nodes = set(self.nodes)
        
        self.leaves = self._list_leaves()
        f.close()
        
    def _find_node(self,id_node):
        ans = -1

        #f = open(self.path_file,"r")
        #lines = f.readlines()
        for k,line in enumerate(self.lines) :
            if str(id_node)==line.split(' [')[0] :
            #if line[0:3] == (str(id_node)+ ' [') :
                ans = k
        return ans
            
    def _check_keyword(self,id_nodes,keyword):
        checks = np.zeros(len(id_nodes))

        #f = open(self.path_file,"r")
        #lines = f.readlines()

        for j,i in enumerate(id_nodes):
            id_l = self._find_node(i) 
            if (' '+keyword) in self.lines[id_l]:
                checks[j] = True
            else:
                checks[j] = False
                
        return checks
            
    def _change_color(self,id_nodes,color_,keyword,out_file=None):
        if out_file is not None:
            f = open(self.path+out_file,"w+")
        else:             
            f = open(self.path_file,"r+")
            
        #f_init = open(self.path_file,"r")
        #lines = f_init.readlines()
        new_lines = self.lines.copy()
        
        checks = self._check_keyword(id_nodes,keyword)
            
        for j,i in enumerate(id_nodes):
            id_l = self._find_node(i)
            
            if checks[j]:
                s_c = new_lines[id_l].split(keyword+'="')[1]
                s_c = s_c.split('"')[0]
                
                new_lines[id_l] = new_lines[id_l].replace(s_c,color_)
            else:
                new_lines[id_l] = new_lines[id_l].replace('] ;',' '+keyword+'="'+color_+'"] ;')

        f.writelines(s + '\n' for s in new_lines)
        #f_init.close()
        #self.lines = f.readlines()
        f.close()
    
    def _add_attribute(self,id_nodes,keyword,value,out_file=None):
        if out_file is not None:
            f = open(self.path+out_file,"w+")
        else:             
            f = open(self.path_file,"r+")
            
        #f_init = open(self.path_file,"r")
        #lines = f_init.readlines()
        new_lines = self.lines.copy()
        for j,i in enumerate(id_nodes):
            id_l = self._find_node(i)

            new_lines[id_l] = new_lines[id_l].replace('] ;',' '+keyword+'='+str(value)+'] ;')

        f.writelines(s + '\n' for s in new_lines)
        #f_init.close()
        #self.lines = f.readlines()
        f.close()     
        
    def _extract_node_info(self,id_):
        attributes = []
        values= []
        
        id_l = self._find_node(id_)
        
        line = self.lines[id_l]
        infos = line.split('label="')[1].split('"')[0]
        list_infos = infos.split('\n')
        
        for l in list_infos:
            attributes.append(l.split(' = ')[0])
            values.append(l.split(' = ')[1])
        return attributes, values
    
    def _write_node_info(self,id_,attr,v):
        f = open(self.path_file,"r+")

        id_l = self._find_node(id_)        
        line = self.lines[id_l]        
        f.close()
        
    def _is_leaf(self,id_):
        
        attr,v = self._extract_node_info(id_)
        if 'X' in attr[0]:
            return False
        else:
            return True
        
    def _list_leaves(self):
        leaves = []
        for n in self.nodes:
            if self._is_leaf(n):
                leaves.append(n)
        return leaves
            
    def _format_leaves(self):
        self._add_attribute(self.leaves,'shape','ellipse',out_file=None)
            
        
    def _change_fillcolor(self,id_nodes,color_,out_file=None):
        if out_file is not None:
            f = open(self.path+out_file,"w+")
        else:             
            f = open(self.path_file,"r+")
            
        f_init = open(self.path_file,"r")
        lines = f_init.readlines()
        new_lines = lines.copy()
        
        checks = self._check_keyword(id_nodes,'fillcolor')
            
        for j,i in enumerate(id_nodes):
            id_l = self._find_node(i)
            
            if checks[j]:
                s_c = new_lines[id_l].split('fillcolor="')[1]
                s_c = s_c.split('"')[0]
                
                new_lines[id_l] = new_lines[id_l].replace(s_c,color_)
            else:
                new_lines[id_l] = new_lines[id_l].replace('] ;',' fillcolor="'+color_+'"] ;')

        f.writelines(s + '\n' for s in new_lines)
        f_init.close()
        f.close()
        
    def _change_edgecolor(self,id_nodes,color_,out_file=None):
        if out_file is not None:
            f = open(self.path+out_file,"w+")
        else:             
            f = open(self.path_file,"r+")
            
        f_init = open(self.path_file,"r")    
        lines = f_init.readlines()
        new_lines = lines.copy()
        checks = self._check_keyword(id_nodes,'color')
            
        for j,i in enumerate(id_nodes):
            id_l = self._find_node(i)
            
            if checks[j]:
                s_c = new_lines[id_l].split('color="')[1]
                s_c = s_c.split('"')[0]
                
                new_lines[id_l] = new_lines[id_l].replace(s_c,color_)
            else:
                new_lines[id_l] = new_lines[id_l].replace('] ;',' color="'+color_+'"] ;')

        f.writelines(s + '\n' for s in new_lines)
        f_init.close()
        f.close()            