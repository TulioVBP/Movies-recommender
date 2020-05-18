import PySimpleGUI as sg
import numpy as np 
import pandas as pd
from scipy.io import loadmat
import scipy.optimize as op
import sqlite3
import time
import hickle as hkl 
# Libraries to download data
import sys
import os
import wget
import hashlib
from zipfile import ZipFile

fonte = ('Helvetica',16)
fonte_tit = ('Helvetica',20)

# SQLite functions
conn = sqlite3.connect('movies.db')
c = conn.cursor()
sqlite3.register_adapter(np.int64, lambda val: int(val))

def create_tables():
    # Input tables
    c.execute('CREATE TABLE IF NOT EXISTS movies (movieId INT, title TEXT, genre TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS genome_scores (genomaId INT, movieId INT, tagId INT, relevance TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS tags (tagId INT, userId INT, movieId INT, tag TEXT, timestamp INT)')
    c.execute('CREATE TABLE IF NOT EXISTS ratings (userId INT,movieId INT, rating REAL,timestamp INT)')
    c.execute('CREATE TABLE IF NOT EXISTS links (movieId INT,imdbId INT, tmdbId INT)')
    c.execute('CREATE TABLE IF NOT EXISTS genome_tags (tagId INT,tag TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS known_users (userId INT, username TEXT)')

def load_data():
    mat_data = loadmat('ex8_movieParams.mat')
    X = np.array(mat_data['X'])
    Theta = np.array(mat_data['Theta'])
    num_users = mat_data['num_users'][0][0]
    num_movies = mat_data['num_movies'][0][0]
    num_features = mat_data['num_features'][0][0]

    mat_data = loadmat('ex8_movies.mat')
    R = np.array(mat_data['R'])
    Y = np.array(mat_data['Y'])

    return X, Theta, num_users, num_movies, num_features, R, Y

def movies_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO movies (movieId, title , genre) VALUES (?,?,?)",row)
    conn.commit()

def genome_scores_entry(values):
    # READ LAST ID
    sql = "SELECT * FROM genome_scores" # SQL query
    df = pd.read_sql_query(sql, conn)
    data = df.values.tolist()
    Id = len(data)
    ii = 0
    for row in values:
        temp = list(row)
        temp.insert(0,Id+ii)
        values[ii] = tuple(temp)
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO genome_scores (genomaId, movieId, tagId, relevance) VALUES (?,?,?,?)",row)
    conn.commit()

def tags_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO tags (userId, movieId, tag, timestamp) VALUES (?,?,?,?)",row)
    conn.commit()

def ratings_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO ratings (userId,movieId, rating,timestamp) VALUES (?,?,?,?)",row)
    conn.commit()

def links_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO links (movieId,imdbId, tmdbId) VALUES (?,?,?)",row)
    conn.commit()

def genome_tags_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO genome_tags (tagId,tag) VALUES (?,?)",row)
    conn.commit()

def known_users_entry(values):
    c.execute("BEGIN TRANSACTION")
    for row in values:
        c.execute("INSERT INTO known_users (userId,username) VALUES (?,?)",row)
    conn.commit()

def read_from_db():
    #sql = "SELECT * FROM expenses" # SQL query
    lt = read_from_db_tablenames()
    print(lt)
    layout = []
    if lt != 'null':
        for ii in range(len(lt)):
            table = lt[ii]
            sql_data = 'PRAGMA table_info(' + table +')' # Query to obtain the tables' data 
            df1 = pd.read_sql_query(sql_data, conn)
            header_list = df1.values[:,1] # Header list
            header_list = header_list.tolist()
            # Getting the data
            sql = "SELECT * FROM " + table # SQL query
            df = pd.read_sql_query(sql, conn)
            data = df.values.tolist()
            layout = layout + [[sg.Text(table.upper(),font=fonte)],[sg.Table(values=data, headings=header_list, display_row_numbers= False,
                            auto_size_columns= True, num_rows=min(25,len(data)),font=fonte)],
                            [sg.Button('Return',font=fonte)]]

        window = sg.Window('Tables', grab_anywhere=False)
        event, values = window.Layout(layout).Read()
        window.Close()
        return data

def read_from_db_tablenames(choice = True):
    sql1 = "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'" # Selecting the table
    df1 = pd.read_sql_query(sql1, conn)
    print(df1.shape)
    tables = df1.values.tolist()
    tables = [row[0] for row in tables]
    tables_CAP = [row[0].upper()+row[1:] for row in tables]
    if choice:
        layout_chooseTable = [[sg.Listbox(values = tables_CAP, size=(30,12),font=fonte)]]
        layout_chooseTable = layout_chooseTable + [[sg.Submit(font=fonte), sg.Cancel(font=fonte)]]
        choose_table = sg.Window('Choose table', grab_anywhere=False)
        event, values = choose_table.Layout(layout_chooseTable).Read()
        choose_table.Close()
    else:
        event = 'Submit'
        values = np.ones(len(tables))
    if event == 'Submit':
        # Instantiate 
        list_of_tables = []
        # Transforming from dictionary to list (if List box, values = {0:['table']})
        values = values[0][0]
        list_of_tables.append(values)
        if list_of_tables == []:
            print('No table was chosen.')
            return
        return [row.lower() for row in list_of_tables]
    else:
        return 'null'
   
def get_movies(movieId):
    nameList = []
    c.execute("BEGIN TRANSACTION")
    for row in movieId:
        A = c.execute("SELECT title FROM movies WHERE movieId = " + str(row))
        name_of_movie = A.fetchall()
        nameList.append(name_of_movie[0][0])
    conn.commit()
    return nameList


# ML functions
def coFi_Cost_Function(params, Y, R, num_users, num_movies,num_features, lambdaV):
    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features]
    X = X.reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:]
    Theta = Theta.reshape(num_users, num_features)
    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    # Compute J and grad
    J  = (1/2* np.power( np.multiply( (X@Theta.T)-Y, R ) , 2 ).sum()  
         + lambdaV / 2 * np.power(Theta,2).sum() + lambdaV / 2 * np.power(X,2).sum() )
    # Grads
    for kk in range(Theta.shape[1]):
        X_grad[:,kk] = (((X @ Theta.T)-Y) * R * Theta[:,kk].T ).sum(axis = 1) + lambdaV*X[:,kk]
        Theta_grad[:,kk] =  ( ((X @ Theta.T)-Y) *R * X[:,kk].reshape(X[:,kk].shape[0],1) ).sum(axis = 0)  + lambdaV*Theta[:,kk]
    # =============================================================
    grad = np.concatenate( (X_grad.ravel(), Theta_grad.ravel()) )

    return J, grad

def sumMat(X,dim):
    sumX = X.sum(axis = dim)
    if dim  == 0: # Sum on rows
        sumX = sumX.reshape(1,sumX.shape[0])
    else:
        sumX = sumX.reshape(sumX.shape[0],1)
    return sumX

def computeNumericalGradient(J, theta):
    # COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    # and gives us a numerical estimate of the gradient.
    #    numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #    gradient of the function J around theta. Calling y = J(theta) should
    #    return the function value at theta.

    #  Notes: The following code implements numerical gradient checking, and 
    #         returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #         approximation of) the partial derivative of J with respect to the 
    #         i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #         be the (approximately) the partial derivative of J with respect 
    #         to theta(i).)
    #                 

    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad

def checkCostFunction(X,Theta,Y,R,lambdaV = 0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem 
    # to check your cost function and gradients
    #   CHECKCOSTFUNCTION(lambdaV) Creates a collaborative filering problem 
    #   to check your cost function and gradients, it will output the 
    #   analytical gradients produced by your code and the numerical gradients 
    #   (computed using computeNumericalGradient). These two gradient 
    #   computations should result in very similar values.
    
    # Check J
    num_users = 4 
    num_movies = 5
    num_features = 3
    X = X[0:num_movies, 0:num_features]
    Theta = Theta[0:num_users, 0:num_features]
    Y = Y[0:num_movies, 0:num_users]
    R = R[0:num_movies, 0:num_users]
    J, grad = coFi_Cost_Function(np.concatenate( (X.ravel(), Theta.ravel()) ), 
           Y, R, num_users, num_movies,num_features, lambdaV)
    print('-------- Test of cost function implementation --------')
    print('Cost function value (should be 22.22):')
    print(J)
    ## Create small problem
    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((5, 3))

    # Zap out most entries
    Y = X_t @ Theta_t.T
    Y[np.random.random(Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.random(X_t.shape)
    Theta = np.random.random(Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    numgrad = computeNumericalGradient(  lambda t: coFi_Cost_Function(t, Y, R, num_users, num_movies,num_features, lambdaV),
                                         np.concatenate((X.ravel(), Theta.ravel())) )

    cost, grad = coFi_Cost_Function(np.concatenate((X.ravel(), Theta.ravel())), Y, R, num_users,
                            num_movies, num_features, lambdaV)
    comp = np.vstack((numgrad,grad)).T
    print(comp)
    print('The above two columns you get should be very similar.\n' 
            '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')
    # diff = norm(numgrad-grad)/norm(numgrad+grad);
    # fprintf(['If your cost function implementation is correct, then \n' ...
    #         'the relative difference will be small (less than 1e-9). \n' ...
    #         '\nRelative Difference: g\n'], diff);

def normalizeRatings(Y, R):
    # %NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
    # %movie (every row)
    # %   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    # %   has a rating of 0 on average, and returns the mean rating in Ymean.
    # %

    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean

def find_params():
    # Collaborative filter
    # 1 - Load data
    # X, Theta, num_users, num_movies, num_features, R ,Y = load_data()
    R, Y = getRY()
    print('R and Y obtained')
    # 2 - Evaluate cost - teste
    #checkCostFunction(X,Theta,Y,R,0)

    # 3 - Find optimal theta and x
    # 3.0 Define smaller Y
    #Y = Y[0:40,4:5]
    #R = R[0:40,4:5]
    # 3.1 - Normalize Y
    Ynorm, Ymean = normalizeRatings(Y, R)

    # 3.2 - Useful values
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    # 3.3 Initial values
    X = np.random.random((num_movies, num_features))
    Theta = np.random.random((num_users, num_features))

    initial_parameters = np.concatenate( (X.ravel(), Theta.ravel()) )
    # 3.4 Optimatimization
    lambdaV = 10
    m,n = X.shape
    options_op = {'maxiter' :100,'disp': True}
    Result = op.fmin_tnc(func = coFi_Cost_Function, 
                                    x0 = initial_parameters, 
                                    args = (Ynorm, R, num_users, num_movies,num_features, lambdaV),
                                    disp = 5,
                                    maxfun= 200)
    optimal_par = Result[0]

    X = optimal_par[0:num_movies*num_features]
    X = X.reshape(num_movies, num_features)
    Theta = optimal_par[num_movies*num_features:]
    Theta = Theta.reshape(num_users, num_features)

    # 4 Store data
    data = { 'X' : X, 'Theta' : Theta, 'Ymean': Ymean }

    # Dump data to file
    hkl.dump(data, 'ml_parameters.hkl' )

    return X, Theta, Ymean

def recommendMovies():
    # Load data from file
    data = hkl.load( 'ml_parameters.hkl' )
    # 1 -Obtaining data
    X = data['X']
    Theta = data['Theta']
    Ymean = data['Ymean']
    
    # 2 - Ask who the user is
    userId = sg.popup_get_text('What is your user ID?',font=fonte)

    # 3 - Get the parameters
    if userId != None:
        idb_userId = c.execute('SELECT DISTINCT userId FROM ratings').fetchall()
        idb_userId = [idb_userId[ii][0] for ii in range(len(idb_userId))]

        idb_movieId = c.execute('SELECT movieId FROM movies').fetchall()
        idb_movieId = [idb_movieId[ii][0] for ii in range(len(idb_movieId))]

        Theta_u = Theta[idb_userId.index(int(userId)),:]
        Theta_u = Theta_u.reshape(len(Theta_u),1)
        pred_score = X @ Theta_u + Ymean
        pred_score = pred_score.flatten()
        sort_index = pred_score.argsort()
        l = len(pred_score)
        #index_top5 = np.argpartition(pred_score, -5)[-5:]
        index_top5 = [np.where(sort_index == l-1-ii) for ii in range(5)]
        top_movies_Id = [idb_movieId[row[0][0]] for row in index_top5]
        name_list = get_movies(top_movies_Id)
        name_vert = ''
        for row in name_list:
            name_vert = name_vert + row + '\n'
        sg.popup_scrolled(name_vert,font=fonte)
       
def getOriginalHash(checksum_raw):
    checksum_raw_str = checksum_raw.decode('utf8')
    checksum_raw_str = checksum_raw_str.split(' ')
    l_word = 0
    checksum = checksum_raw_str[0]
    # Update checksum to take the longest word
    for word in checksum_raw_str:
        if len(word)>l_word:
            checksum = word
            l_word = len(word)
    # Getting rid of \n
    checksum = checksum.replace('\n', '')
    print('\n')
    print(checksum)
    return checksum

def download_data():
    path = os.getcwd() # Getting current working directory 
    print("The current directory is " + path)
    # 0 - Ask which data to download
    layout = [[sg.Text('Which repository do you fancy?',font=fonte)],
                 [sg.Button('250 MB',font=fonte,key=250),sg.Button('100 KB',font=fonte,key=1)],
                 [sg.Button('Return',font=fonte)]]
    
    file_dict = {250:'ml-25m.zip', 
                 1:'ml-latest-small.zip'}
    md5_dict = {250:'ml-25m.zip.md5',
                1:False}
    url_dict = {250:'http://files.grouplens.org/datasets/movielens/ml-25m.zip', 
                1:'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'}
    
    button,values = sg.Window('Dataset choice',layout=layout).read()
    window.Close()
    if button != 'Return':
        print(button)
        filename = file_dict[button]
        CSname = md5_dict[button]
        # 1 - Download dataset (first check if it was already downloaded)
        try:
            f = open(filename,"rb")
            bytes = f.read() # read file as bytes
            hash_file = hashlib.md5(bytes).hexdigest()
            print("Hash of downloaded file: " + hash_file)
        except IOError:
            url = url_dict[button]
            wget.download(url, path, bar = bar_progress)
            f = open(filename,"rb")
            bytes = f.read() # read file as bytes
            hash_file = hashlib.md5(bytes).hexdigest()
            print("Hash of downloaded file: " + hash_file)
        finally:
            f.close()

        # 2 - Download checksum
        url_dict_md5 = {250:'http://files.grouplens.org/datasets/movielens/ml-25m.zip.md5',
                        1:'http://files.grouplens.org/datasets/movielens/ml-100k.zip.md5'}
        if md5_dict[button]:
            try:
                f = open(CSname,"rb")
                checksum_raw = f.read()
                checksum = getOriginalHash(checksum_raw)
                print("Original file hash: " + checksum)
            except IOError:
                url = url_dict_md5[button]
                wget.download(url, path, bar = bar_progress)
                f = open(CSname,"rb")
                checksum_raw = f.read()
                checksum = getOriginalHash(checksum_raw)
            finally:
                f.close()

            # 3 - Check parity

            if hash_file == checksum:
                print('\n File was correctly downloaded. Files contained in the zip file:')
                unzipFile(filename)
            else:
                print('Error! Download the file again.')
        else:
            unzipFile(filename)

def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def unzipFile(filename):
    with ZipFile(filename, 'r') as zipObj:
        # Get list of files names in zip
        listOfiles = zipObj.namelist()
        # Iterate over the list of file names in given list & print them
        for elem in listOfiles:
            print(elem)
        # Extract all
        path = os.getcwd()
        buttons = 'Yes'
        if os.path.isdir(path +'/' +listOfiles[0]):
            #File already downloaded
            buttons = sg.popup_yes_no('The file you are about to unzip already exists. Do you want to continue and overwrite the current files?',font=fonte)
            #sg.EasyPrint(buttons)
        if buttons == 'Yes':
            zipObj.extractall(path=path)
        zipObj.close()
        
def loadTables():  
     # 0 - Ask which data to download
    layout = [[sg.Text('Which repository do you fancy?',font=fonte)],
                 [sg.Button('250 MB',font=fonte,key=250),sg.Button('100 KB',font=fonte,key=1)],
                 [sg.Button('Return',font=fonte)]] 
    button,values = sg.Window('Dataset choice',layout=layout).read()
    window.Close()
    path_dic = {250:'/ml-25m',1:'/ml-latest-small'}
    pathFolder = os.getcwd()+path_dic[button]
    tables_names = ('movies.csv','ratings.csv','tags.csv','links.csv')#'genome-tags.csv','genome-scores.csv')
    dict_fun = {0:movies_entry, 1:ratings_entry, 2:tags_entry, 3:links_entry, 4:genome_tags_entry, 5:genome_scores_entry}
    # Movies
    for jj in range(len(tables_names)):
        df = pd.read_csv(pathFolder+'/'+tables_names[jj])
        rows = df.index
        value = [tuple(df.loc[ii,:]) for ii in rows]
        dict_fun[jj](value)
        print("Table"+tables_names[jj] +"loaded.")
    sg.popup_ok('All tables correctly loaded!')

def delete_all():
    sure = sg.popup_yes_no("Continue will erase your whole database. Are you sure you want to continue? ", font=fonte)
    if sure == 'Yes':
        sql1 = "SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'" # Selecting the table
        df1 = pd.read_sql_query(sql1, conn)
        tables = df1.values.tolist()
        tables = [row[0] for row in tables]
        for ii in range(len(tables)):
            sql_query = 'DELETE FROM '+tables[ii] 
            c.execute(sql_query)
            conn.commit()
        sg.popup_ok('Deleted.',font=fonte)

def getRY():
    # 0 - Count number of user and movies
    num_users = c.execute('SELECT COUNT(DISTINCT userId) FROM ratings')
    num_users = num_users.fetchall()[0][0]
    print(num_users)
    num_movies = c.execute('SELECT COUNT(DISTINCT movieId) FROM movies')
    num_movies = num_movies.fetchall()[0][0]
    print(num_movies)

    # 1 - Instatiate R and Y
    R = np.zeros((num_movies,num_users))
    Y = np.zeros((num_movies,num_users))

    # 2 - Update movies and ratings
    idb_movieId = c.execute('SELECT movieId FROM movies').fetchall()
    idb_movieId = [idb_movieId[ii][0] for ii in range(len(idb_movieId))]

    idb_userId = c.execute('SELECT DISTINCT userId FROM ratings').fetchall()
    idb_userId = [idb_userId[ii][0] for ii in range(len(idb_userId))]
    c.execute("BEGIN TRANSACTION")
    for ii in range(num_users):
        kk = idb_userId[ii] 
        A = c.execute('SELECT movieId,rating FROM ratings WHERE userId = '+ str(kk)) # +1 to find the one
        A = A.fetchall()
        movies_id_per_user = [A[jj][0] for jj in range(len(A))] # -1 to start the index from 0
        movies_index = [idb_movieId.index(row) for row in movies_id_per_user]
        ratings_val = [A[jj][1] for jj in range(len(A))]
        R[movies_index, ii] = 1
        Y[movies_index, ii] = ratings_val
    # 3 - Return data
    conn.commit()
    return R,Y

def get_user_avaliations(name, movieId, userId):
    layout_GUA = [[sg.Text("How do you rate " + name + "?",font=fonte)],
                  [sg.DropDown([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],font=fonte)]
                  [sg.Button('Rate',key = True),sg.Button('Return', key = False)]]
    
    # Create window
    window = sg.Window('Rate a movie',layout = layout_GUA)
    button, values = window.Read()
    
    rate = False
    if button:
        rating = value
        timestamp = time.time()
        ratings_entry((userId, movieId, rating, timestamp))

# ------------- PySimpleGUI functions
def manipulate_data():
    layout_MD = [[sg.Text("What do you want here?!")],
                [sg.Button('1 - Download data',font = fonte, key=1)],
                [sg.Button('5 - Load data into table', font = fonte, key=2)],
                [sg.Button('1 - Read tables',font = fonte,key=3)],
                [sg.Button('2 - Delete tables',font = fonte,key=4)],
                [sg.Cancel(key='C', font=fonte) ]]
    
    fun_dict = {1:download_data, 2:loadTables, 3: read_from_db, 4: delete_all}
    
    window = sg.Window('Intruser',layout=layout_MD)
    button, values = window.Read()
    if button != None:
        if button != 'C':
            fun_dict[button]()

def registered_users():
    sql_data = 'PRAGMA table_info( known_users )' # Query to obtain the tables' data 
    df1 = pd.read_sql_query(sql_data, conn)
    header_list = df1.values[:,1] # Header list
    header_list = header_list.tolist()
    # Getting the data
    sql = "SELECT * FROM known_users" # SQL query
    df = pd.read_sql_query(sql, conn)
    data = df.values.tolist()
    if len(data)>0:
        layout = [[sg.Text("Registered users",font=fonte)],[sg.Table(values=data, headings=header_list, display_row_numbers= False,
                            auto_size_columns= True, num_rows=min(25,len(data)),font=fonte)],
                            [sg.Button('Add an user',font=fonte,key = 'A') , sg.Button('Return',font=fonte)]]
    else:
        layout = [[sg.Button('Add an user',font=fonte,key = 'A') , sg.Button('Return',font=fonte)]]
    
    window = sg.Window('Tables', grab_anywhere=False)
    event, values = window.Layout(layout).Read()
    window.Close()

    if event == 'A':
        username = sg.popup_get_text('What is the name of the new user?',font=fonte)
        A = c.execute('SELECT DISTINCT userId FROM ratings').fetchall()

        A = [A[ii][0] for ii in range(len(A))]
        last_userId = max(A)
        if len(data)>0:
            B = c.execute('SELECT DISTINCT userId FROM known_users').fetchall()
            B = [B[ii][0] for ii in range(len(B))]
            last_userIdB = max(B)
            if max(B) > max(A):
                last_userId = last_userIdB
        conn.commit()

        userId = last_userId + 1

        known_users_entry([(userId,username)])
        sg.popup_scrolled('Your user ID is ' + str(userId),font=fonte)
    


class DefaultKeyDict(dict):
    def __init__(self, default_key, *args, **kwargs):
        self.default_key = default_key
        super(DefaultKeyDict, self).__init__(*args, **kwargs)

    def __missing__ (self, key):
        if self.default_key not in self:  # default key not defined
            raise KeyError(key)
        return self[self.default_key]

    def __repr__(self):
        return ('{}({!r}, {})'.format(self.__class__.__name__,
                                      self.default_key,
                                      super(DefaultKeyDict, self).__repr__()))

    def __reduce__(self):  # optional, for pickle support
        args = (self.default_key if self.default_key in self else None,)
        return self.__class__, args, None, None, self.iteritems()

# ----------------------- MAIN ----------------------
create_tables()

# 1 - Define main


# 2 - Run menu
fun_dict = DefaultKeyDict(0 ,{0:print, 2:find_params, 3:recommendMovies , 4:registered_users , 5: manipulate_data})

while True:
    layout_main = [[sg.Text('Movie prediction',font=fonte_tit)],
                   [sg.Button('1 - Rate movies',font = fonte,key=1)],
                    [sg.Button('2 - Train model',font = fonte,key=2)],
                    [sg.Button('3 - Recommend movies',font=fonte,key=3)],
                    [sg.Button('4 - Registered users',font=fonte,key=4)],
                    [sg.Button('5 - Manipulate tables',font = fonte, key = 5)],
                    [sg.Cancel(font = fonte)]]
    window = sg.Window('Movie prediction',layout=layout_main)
    button, value = window.Read()
    window.Close()
    fun_dict[button]()
    if button is None:
        break

c.close()
conn.close()