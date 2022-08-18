import pandas as pd
from typing import Dict, Pattern
import os, re, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold # 分层分割
char_to_remove = ['+','-','*','/','=','++','--','\\','<str>','<char>','|','&','!']
class MyDataset:
    def __init__(self,
                all_releases:dict,
                data_root_dir = '../datasets/original/',
                # save_dir = "../datasets/preprocessed_data/",
                test_size=0.2
                ):
    
        self.all_releases=all_releases
        self.data_root_dir=data_root_dir        
        self.test_size=test_size

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # self.save_dir=save_dir
        self.file_lvl_dir = os.path.join(data_root_dir,'File-level')
        self.line_lvl_dir = os.path.join(data_root_dir,'Line-level')
    def preprocess_data(self,proj_name):
        cur_all_rel = self.all_releases[proj_name]
        result={}
        for rel in cur_all_rel:
            result[rel]=self.get_rel_data(rel)            
        return result
    def get_cp_file_data_split(self):
        train_list=[]
        eval_list=[]
        test_list=[]        
        for proj_name in self.all_releases.keys():
            train,eval,test=self.get_file_data_split(proj_name)
            train_list.append(train)
            eval_list.append(eval)
            test_list.append(test)
        train_set=pd.concat(train_list,ignore_index=True)
        eval_set=pd.concat(eval_list,ignore_index=True)
        test_set=pd.concat(test_list,ignore_index=True)
        return train_set,eval_set,test_set
    def get_file_data_split(self,proj_name):
        cur_all_rel = self.all_releases[proj_name]
        df_list=[]
        for rel in cur_all_rel:
            df=pd.read_csv(os.path.join(self.file_lvl_dir,rel+'_ground-truth-files_dataset.csv'), encoding='latin')
            df['Project']=proj_name
            df['Release']=rel
            df['RelFilename']=df['Release']+'$'+df['File']
            df_list.append(df)
        all_data=pd.concat(df_list,ignore_index=True)
        all_data['target']=all_data['Bug'].astype(int)
        # train_eval = all_data.iloc[int(len(all_data)*self.test_size):].reset_index(drop = True)
        # test  = all_data.iloc[:int(len(all_data)*self.test_size)].reset_index(drop = True)    
        # train = train_eval.iloc[int(len(train_eval)*self.test_size):].reset_index(drop = True)
        # eval = train_eval.iloc[:int(len(train_eval)*self.test_size)].reset_index(drop = True)
        skf = StratifiedKFold(n_splits=5)
        t=all_data.Bug
        ids=[]
        for train_index, test_index in skf.split(np.zeros(len(t)), t):
            ids.append((train_index, test_index))
        train_index, test_index=ids[0]
        train_eval = all_data.loc[train_index].reset_index(drop=True)
        test = all_data.loc[test_index].reset_index(drop=True)
       
        t=train_eval.Bug
        ids=[]
        for train_index, test_index in skf.split(np.zeros(len(t)), t):
            ids.append((train_index, test_index))
        train_index, test_index=ids[0]
        train = train_eval.loc[train_index].reset_index(drop=True)
        eval = train_eval.loc[test_index].reset_index(drop=True)
        return train,eval,test
    def get_rel_data(self,rel):
        file_level_data = pd.read_csv(os.path.join(self.file_lvl_dir,rel+'_ground-truth-files_dataset.csv'), encoding='latin')
        line_level_data = pd.read_csv(os.path.join(self.line_lvl_dir,rel+'_defective_lines_dataset.csv'), encoding='latin')        
        buggy_files = list(line_level_data['File'].unique())

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():
            
            filename = row['File']

            if '.java' not in filename:
                continue

            code = row['SRC']
            label = row['Bug']

            code_df = create_code_df(code, filename)
            code_df['file-label'] = [label]*len(code_df)
            code_df['line-label'] = [False]*len(code_df)

            if filename in buggy_files:
                buggy_lines = list(line_level_data[line_level_data['File']==filename]['Line_number'])
                code_df['line-label'] = code_df['line_number'].isin(buggy_lines)

            if len(code_df) > 0:
                preprocessed_df_list.append(code_df)

        return pd.concat(preprocessed_df_list)
    def get_dataset(self):
        for proj in list(self.all_releases.keys()):
            self.preprocess_data(proj)


def is_comment_line(code_line, comments_list):
    '''
        input
            code_line (string): source code in a line
            comments_list (list): a list that contains every comments
        output
            boolean value
    '''

    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('//'):
        return True
    elif code_line in comments_list:
        return True
    
    return False

def is_empty_line(code_line):
    '''
        input
            code_line (string)
        output
            boolean value
    '''

    if len(code_line.strip()) == 0:
        return True

    return False

def preprocess_code_line(code_line):
    '''
        input
            code_line (string)
    '''

    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub('\b\d+\b','',code_line)
    code_line = re.sub("\\[.*?\\]", '', code_line)
    code_line = re.sub("[\\.|,|:|;|{|}|(|)]", ' ', code_line)

    for char in char_to_remove:
        code_line = code_line.replace(char,' ')

    code_line = code_line.strip()

    return code_line

def create_code_df(code_str, filename):
    '''
        input
            code_str (string): a source code
            filename (string): a file name of source code

        output
            code_df (DataFrame): a dataframe of source code that contains the following columns
            - code_line (str): source code in a line
            - line_number (str): line number of source code line
            - is_comment (bool): boolean which indicates if a line is comment
            - is_blank_line(bool): boolean which indicates if a line is blank
    '''

    df = pd.DataFrame()

    code_lines = code_str.splitlines()
    
    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []


    comments = re.findall(r'(/\*[\s\S]*?\*/)',code_str,re.DOTALL)
    comments_str = '\n'.join(comments)
    comments_list = comments_str.split('\n')

    for l in code_lines:
        l = l.strip()
        is_comment = is_comment_line(l,comments_list)
        is_comments.append(is_comment)
        # preprocess code here then check empty line...

        if not is_comment:
            l = preprocess_code_line(l)
            
        is_blank_line.append(is_empty_line(l))
        preprocess_code_lines.append(l)

    if 'test' in filename:
        is_test = True
    else:
        is_test = False

    df['filename'] = [filename]*len(code_lines)
    df['is_test_file'] = [is_test]*len(code_lines)
    df['code_line'] = preprocess_code_lines
    df['line_number'] = np.arange(1,len(code_lines)+1)
    df['is_comment'] = is_comments
    df['is_blank'] = is_blank_line

    return df
class MethodExtractor():
    def __init__(self):
        self.regex_patters = {
            "JAVA":       r"\b(?!if|for|while|switch|catch)\b[a-zA-Z\d_]+?\s*?\([a-zA-Z\d\s_,\>\<\?\*\.\[\]]*?\)\s*?\{",
            "KOTLIN":     r"fun\s[a-zA-Z\d_\.]+?\s*?\([a-zA-Z\d\s_,\?\@\>\<\?\*\.\[\]\:]*?\)\s*?.*?(\{|\=)",
            "OBJC":       r"[\-\+]\s*?[a-zA-Z\d_\(\)\:\*\s]+?\s*?\{",
            "SWIFT":      r"func\s*?[a-zA-Z\d_\(\)\:\*\s\-\<\>\?\,\[\]\.]+?\s*?\{",
            "RUBY":       r"(def)\s(.+)",
            "GROOVY":     r"\b(?!if|for|while|switch|catch)\b[a-zA-Z\d_]+?\s*?\([a-zA-Z\d\s_,\>\<\?\*\.\[\]\=\@\']*?\)\s*?\{",
            "JAVASCRIPT": r"(function\s+?)([a-zA-Z\d_\:\*\-\<\>\?\,\[\]\.\s\|\=\$]+?)\(([a-zA-Z\d_\(\)\:\*\s\-\<\>\?\,\[\]\.\|\=\$\/]*?)\)*?[\:]*?\s*?\{",
            "TYPESCRIPT": r"(function\s+?)([a-zA-Z\d_\:\*\-\<\>\?\,\[\]\.\s\|\=\$]+?)\(([a-zA-Z\d_\(\)\:\*\s\-\<\>\?\,\[\]\.\|\=\$\/]*?)\)*?[\:]*?\s*?\{",
            "C":          r"\b(?!if|for|while|switch)\b[a-zA-Z\d_]+?\s*?\([a-zA-Z\d\s_,\*]*?\)\s*?\{",
            "CPP":        r"\b(?!if|for|while|switch)\b[a-zA-Z\d\_\:\<\>\*\&]+?\s*?\([\(a-zA-Z\d\s_,\*&:]*?\)\s*?\w+\s*?\{",
            "PY":         r"(def)\s.+(.+):"
        }
        self.compiled_re: Dict[str, Pattern] = {}
        self._compile()
    def _compile(self):
        for name, pattern in self.regex_patters.items():
            self.compiled_re[name] = re.compile(pattern)
    def __get_expression(self, language:str) -> Pattern:
        return self.compiled_re[language.upper()]
    def extract(self,src:str,language:str):
        find_method_expression = self.__get_expression(language)
        return find_method_expression.findall(src)


if __name__ == "__main__":
    all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'], 
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'], 
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'], 
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'], 
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'], 
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'], 
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'], 
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}
    # data_dir=r"D:\data_sci\line-level-defect-prediction\Dataset"
    data_dir=r"/home/hickey/data/line-level-defect-prediction/Dataset"
    md=MyDataset(all_releases,data_dir)
    
    train,eval,test=md.get_file_data_split('groovy')
    src=eval.iloc[1]['SRC']
    me=MethodExtractor()
    # res=me.extract(src,"JAVA")
    # print(res)
    # me.extract_java(src)
    from transformers import T5Tokenizer
    tokenizer_name=r"/home/hickey/python-workspace/LineVul/linevul/saved_models/razent/cotext-1-ccg"

    tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    tokens=tokenizer.tokenize(src)
    print(tokens)