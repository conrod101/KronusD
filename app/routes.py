from app import app
from app import db
import os
import urllib.request
import cv2
import pickle
import imutils
import sklearn
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask import render_template, request, redirect, url_for, flash, Flask
from app.forms import LoginForm
from flask_login import login_user, logout_user, current_user, login_required
from app.models import User
from flask import request
from werkzeug.urls import url_parse
from datetime import datetime
import sqlite3
from flask_bootstrap import Bootstrap
from tensorflow.keras.applications.vgg16 import preprocess_input
from app import app
from flask import render_template, request, redirect, url_for, flash
from app.forms import LoginForm
from app.forms import RegistrationForm

covid_model = load_model('models/covid.h5')
braintumor_model = load_model('models/braintumor.h5')
alzheimer_model = load_model('models/alzheimer_model.h5')
diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
pneumonia_model = load_model('models/pneumonia_model.h5')
breastcancer_model = joblib.load('models/cancer_model.pkl')
# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

########################### Routing Functions ########################################
###### sqlite 3 code #######

conn = sqlite3.connect('database/lifecare.db',check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS doctors(
            first_name text,
            last_name text,
            dob date,
            phone_number integer,
            address integer text,
            doc_id integer text,
            password integer text,
            speciality text,
            status integer
            )''')

c.execute('''CREATE TABLE IF NOT EXISTS patients(
            first_name text,
            last_name text,
            dob date,
            phone_number integer,
            password integer text,
            address integer text,
            status integer
            )''')

c.execute('''CREATE TABLE IF NOT EXISTS superusercreds(
            username integer text,
            password integer text
            )''')

c.execute('''CREATE TABLE IF NOT EXISTS doctorappointmentrequests(
            docid integer text,
            patientname integer text,
            patientnum integer text,
            appointmentdate date
            )''')

c.execute('''CREATE TABLE IF NOT EXISTS doctorappointments(
            docid integer text,
            patientname integer text,
            patientnum integer text,
            appointmentdate date
            )''')


###############################
c.execute('SELECT * from superusercreds')
conn.commit()
adminuser = c.fetchall()
if not adminuser:
    c.execute("INSERT INTO superusercreds VALUES ('admin','admin')")
    conn.commit()

# c.execute('SELECT * FROM id_and_password')
# c.execute("INSERT INTO id_and_password VALUES ('{}','{}','{}','{}','{}','{}')".format(ent3.get(), ent4.get(),ent5.get(), ent0.get(),ent1.get(), ent2.get()))
# c1.execute('DELETE FROM count')
# c.execute("UPDATE id_and_password SET user_id=(?) WHERE user_id=(?)",(eee4.get(),c3[0][0]))

###########################
def datetoday():
    today = datetime.now().strftime("%Y-%m-%d")
    return today

def checkonlyalpha(x):
    return x.isalpha()

def checkonlynum(x):
    return x.isdigit()

def checkpass(x):
    f1,f2,f3 = False,False,False
    if len(x)<8:
        return False
    if any(c.isalpha() for c in x):
        f1 = True
    if any(c.isdigit() for c in x):
        f2 = True
    if any(c in x for c in ['@','$','!']):
        f3 = True

    return f1 and f2 and f3

def checkphnlen(x):
    return len(x)==10

def retalldocsandapps():
    c.execute(f"SELECT * FROM doctorappointments")
    conn.commit()
    docsandapps = c.fetchall()
    l = len(docsandapps)
    return docsandapps,l

def getpatdetails(phn):
    c.execute(f"SELECT * FROM patients")
    conn.commit()
    patients = c.fetchall()
    for i in patients:
        if str(i[3])==str(phn):
            return i

def getdocdetails(docid):
    c.execute(f"SELECT * FROM doctors")
    conn.commit()
    doctors = c.fetchall()
    for i in doctors:
        if str(i[5])==str(docid):
            return i


def retdocsandapps(docid):
    c.execute(f"SELECT * FROM doctorappointments")
    conn.commit()
    docsandapps = c.fetchall()
    docsandapps2 = []
    for i in docsandapps:
        if str(i[0])==str(docid):
            docsandapps2.append(i)
    l = len(docsandapps2)
    return docsandapps2,l

def retapprequests(docid):
    c.execute(f"SELECT * FROM doctorappointmentrequests")
    conn.commit()
    appreq = c.fetchall()
    appreq2 = []
    for i in appreq:
        if str(i[0])==str(docid):
            appreq2.append(i)
    l = len(appreq2)
    return appreq,l

def ret_patient_reg_requests():
    c.execute('SELECT * FROM patients')
    conn.commit()
    data = c.fetchall()
    patient_reg_requests = []
    for d in data:
        if str(d[-1])=='0':
            patient_reg_requests.append(d)
    return patient_reg_requests

def ret_doctor_reg_requests():
    c.execute('SELECT * FROM doctors')
    conn.commit()
    data = c.fetchall()
    doctor_reg_requests = []
    for d in data:
        if str(d[-1])=='0':
            doctor_reg_requests.append(d)
    return doctor_reg_requests

def ret_registered_patients():
    c.execute('SELECT * FROM patients')
    conn.commit()
    data = c.fetchall()
    registered_patients = []
    for d in data:
        if str(d[-1])=='1':
            registered_patients.append(d)
    return registered_patients

def ret_registered_doctors():
    c.execute('SELECT * FROM doctors')
    conn.commit()
    data = c.fetchall()
    registered_doctors = []
    for d in data:
        if str(d[-1])=='1':
            registered_doctors.append(d)
    return registered_doctors

def ret_docname_docspec():
    c.execute('SELECT * FROM doctors')
    conn.commit()
    registered_doctors = c.fetchall()
    docname_docid = []
    for i in registered_doctors:
        docname_docid.append(str(i[0])+' '+str(i[1])+'-'+str(i[5])+'-'+str(i[7]))
    l = len(docname_docid)
    return docname_docid,l

def getdocname(docid):
    c.execute('SELECT * FROM doctors')
    conn.commit()
    registered_doctors = c.fetchall()
    for i in registered_doctors:
        if str(i[5])==str(docid):
            return i[0]+'-'+i[1]

def getpatname(patnum):
    c.execute('SELECT * FROM patients')
    conn.commit()
    details = c.fetchall()
    for i in details:
        if str(i[3])==str(patnum):
            return i[0]+' '+i[1]
    else:
        return -1

def get_all_docids():
    c.execute('SELECT * FROM doctors')
    conn.commit()
    registered_doctors = c.fetchall()
    docids = []
    for i in registered_doctors:
        docids.append(str(i[5]))
    return docids


def get_all_patnums():
    c.execute('SELECT * FROM patients')
    conn.commit()
    registered_patients = c.fetchall()
    patnums = []
    for i in registered_patients:
        patnums.append(str(i[3]))
    return patnums


################## ROUTING FUNCTIONS #########################

#### Routing for your application###


# @app.route('/')
# @app.route('/index')
# def index():

    
#     return render_template('index.html')
@app.route('/')
@app.route('/index')
def index():

    return render_template('index.html', title='Home')

@app.route('/dashboard')
# @login_required
def dashboard():

    return render_template('dashboard.html')

@app.route('/home')
def home():
    """Render website's home page."""
    return render_template('index.html')


@app.route('/homepage')
def homepage():
    """Render website's home page."""
    return render_template('homepage.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('dashboard')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # form = RegisterForm()
    # if form.validate_on_submit():
    #     hashed_password = generate_password_hash(form.password.data, method='sha256')
    #     new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
    #     db.session.add(new_user)
    #     db.session.commit()

        # return redirect("/login")
    return render_template('signup.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/about/')
def about():
    """Render the website's about page."""
    return render_template('about.html', name="Mary Jane")

@app.route('/help')
def help():
    """Render the website's about page."""
    return render_template('help.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


########################### Result Functions ########################################


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = crop_imgs([img])
            img = img.reshape(img.shape[1:])
            img = preprocess_imgs([img], (224, 224))
            pred = braintumor_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Brain Tumor test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
            return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        pred = diabetes_model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        pred = breastcancer_model.predict(
            np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour BReast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/resulta', methods=['GET', 'POST'])
def resulta():
    if request.method == 'POST':
        print(request.url)
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3)
            img = img/255.0
            pred = alzheimer_model.predict(img)
            pred = pred[0].argmax()
            print(pred)
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Alzheimer test results are ready.\nRESULT: {}'.format(firstname,['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'][pred]))
            return render_template('resulta.html', filename=filename, fn=firstname, ln=lastname, age=age, r=0, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect('/')


@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


@app.route('/homee')
def homee():
    return render_template('homee.html') 


@app.route('/patreg')
def patreg():
    return render_template('patientregistration.html') 


@app.route('/docreg')
def docreg():
    return render_template('doctorregistration.html') 


@app.route('/loginpage1')
def loginpage1():
    return render_template('loginpage1.html') 


@app.route('/loginpage2')
def loginpage2():
    return render_template('loginpage2.html') 


@app.route('/loginpage3')
def loginpage3():
    return render_template('loginpage3.html') 


### Functions for adding Patients
@app.route('/addpatient',methods=['POST'])
def addpatient():
    passw = request.form['password']
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    dob = request.form['dob']
    phn = request.form['phn']
    address = request.form['address']
    print(firstname,lastname,checkonlyalpha(firstname),checkonlyalpha(lastname))

    if (not checkonlyalpha(firstname)) | (not checkonlyalpha(lastname)):
        return render_template('home.html',mess=f'First Name and Last Name can only contain alphabets.')

    if not checkpass(passw):
        return render_template('home.html',mess=f"Password should be of length 8 and should contain alphabets, numbers and special characters ('@','$','!').") 

    if not checkphnlen(phn):
        return render_template('home.html',mess=f"Phone number should be of length 10.") 

    if str(phn) in get_all_patnums():
        return render_template('home.html',mess=f'Patient with mobile number {phn} already exists.') 
    c.execute(f"INSERT INTO patients VALUES ('{firstname}','{lastname}','{dob}','{phn}','{passw}','{address}',0)")
    conn.commit()
    return render_template('home.html',mess=f'Registration Request sent to Super Admin for Patient {firstname}.') 


### Functions for adding Doctors
@app.route('/adddoctor',methods=['GET','POST'])
def adddoctor():
    passw = request.form['password']
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    dob = request.form['dob']
    phn = request.form['phn']
    address = request.form['address']
    docid = request.form['docid']
    spec = request.form['speciality']
    
    if not checkonlyalpha(firstname) and not checkonlyalpha(lastname):
        return render_template('home.html',mess=f'First Name and Last Name can only contain alphabets.')

    if not checkonlyalpha(spec):
        return render_template('home.html',mess=f'Doctor Speciality can only contain alphabets.')

    if not checkpass(passw):
        return render_template('home.html',mess=f"Password should be of length 8 and should contain alphabets, numbers and special characters ('@','$','!').") 
    
    if not checkphnlen(phn):
        return render_template('home.html',mess=f"Phone number should be of length 10.") 

    if str(docid) in get_all_docids():
        return render_template('home.html',mess=f'Doctor with Doc ID {docid} already exists.') 
    c.execute(f"INSERT INTO doctors VALUES ('{firstname}','{lastname}','{dob}','{phn}','{address}','{docid}','{passw}','{spec}',0)")
    conn.commit()
    return render_template('home.html',mess=f'Registration Request sent to Super Admin for Doctor {firstname}.') 


## Patient Login Page
@app.route('/patientlogin',methods=['GET','POST'])
def patientlogin():
    phn = request.form['phn']
    passw = request.form['pass']
    c.execute('SELECT * FROM patients')
    conn.commit()
    registerd_patients = c.fetchall()
    for i in registerd_patients:
        if str(i[3])==str(phn) and str(i[4])==str(passw):
            docsandapps,l = retalldocsandapps()
            docname_docid,l2 = ret_docname_docspec()
            docnames = []
            for i in docsandapps:
                docnames.append(getdocname(i[0]))
            return render_template('patientlogin.html',docsandapps=docsandapps,docnames=docnames,docname_docid=docname_docid,l=l,l2=l2,patname=i[0],phn=phn) 

    else:
        return render_template('loginpage1.html',err='Please enter correct credentials...')


## Doctor Login Page
@app.route('/doctorlogin',methods=['GET','POST'])
def doctorlogin():
    docid = request.form['docid']
    passw = request.form['pass']
    c.execute('SELECT * FROM doctors')
    conn.commit()
    registerd_doctors = c.fetchall()

    for i in registerd_doctors:
        if str(i[5])==str(docid) and str(i[6])==str(passw):
            appointment_requests_for_this_doctor,l1 = retapprequests(docid)
            fix_appointment_for_this_doctor,l2 = retdocsandapps(docid)
            return render_template('doctorlogin.html',appointment_requests_for_this_doctor=appointment_requests_for_this_doctor,fix_appointment_for_this_doctor=fix_appointment_for_this_doctor,l1=l1,l2=l2,docname=i[0],docid=docid)

    else:
        return render_template('loginpage2.html',err='Please enter correct credentials...')
    

## Admin Login Page
@app.route('/adminlogin',methods=['GET','POST'])
def adminlogin():
    username = request.form['username']
    passw = request.form['pass']
    c.execute('SELECT * FROM superusercreds')
    conn.commit()
    superusercreds = c.fetchall()

    for i in superusercreds:
        if str(i[0])==str(username) and str(i[1])==str(passw):
            patient_reg_requests = ret_patient_reg_requests()
            doctor_reg_requests = ret_doctor_reg_requests()
            registered_patients = ret_registered_patients()
            registered_doctors = ret_registered_doctors()
            l1 = len(patient_reg_requests)
            l2 = len(doctor_reg_requests)
            l3 = len(registered_patients)
            l4 = len(registered_doctors)
            return render_template('adminlogin.html',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4)
    else:
        return render_template('loginpage3.html',err='Please enter correct credentials...')
    

## Delete patient from database    
@app.route('/deletepatient',methods=['GET','POST'])
def deletepatient():
    patnum = request.values['patnum']
    c.execute(f"DELETE FROM patients WHERE phone_number='{str(patnum)}' ")
    conn.commit()
    patient_reg_requests = ret_patient_reg_requests()
    doctor_reg_requests = ret_doctor_reg_requests()
    registered_patients = ret_registered_patients()
    registered_doctors = ret_registered_doctors()
    l1 = len(patient_reg_requests)
    l2 = len(doctor_reg_requests)
    l3 = len(registered_patients)
    l4 = len(registered_doctors)
    return render_template('adminlogin.html',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4)
    

## Delete doctor from database
@app.route('/deletedoctor',methods=['GET','POST'])
def deletedoctor():
    docid = request.values['docid']
    c.execute(f"DELETE FROM doctors WHERE doc_id='{str(docid)}' ")
    conn.commit()
    patient_reg_requests = ret_patient_reg_requests()
    doctor_reg_requests = ret_doctor_reg_requests()
    registered_patients = ret_registered_patients()
    registered_doctors = ret_registered_doctors()
    l1 = len(patient_reg_requests)
    l2 = len(doctor_reg_requests)
    l3 = len(registered_patients)
    l4 = len(registered_doctors)
    return render_template('adminlogin.html',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4)
   

## Patient Function to make appointment
@app.route('/makeappointment',methods=['GET','POST'])
def makeappointment():
    patnum = request.args['phn']
    appdate = request.form['appdate']
    whichdoctor = request.form['whichdoctor']
    docname = whichdoctor.split('-')[0]
    docid = whichdoctor.split('-')[1]
    patname = getpatname(patnum)
    appdate2 = datetime.strptime(appdate, '%Y-%m-%d').strftime("%Y-%m-%d")
    print(appdate2,datetoday())
    if appdate2>datetoday():
        if patname!=-1:
            c.execute(f"INSERT INTO doctorappointmentrequests VALUES ('{docid}','{patname}','{patnum}','{appdate}')")
            conn.commit()
            docsandapps,l = retalldocsandapps()
            docname_docid,l2 = ret_docname_docspec()
            docnames = []
            for i in docsandapps:
                docnames.append(getdocname(i[0]))
            return render_template('patientlogin.html',mess=f'Appointment Request sent to doctor.',docnames=docnames,docsandapps=docsandapps,docname_docid=docname_docid,l=l,l2=l2,patname=patname) 
        else:
            docsandapps,l = retalldocsandapps()
            docname_docid,l2 = ret_docname_docspec()
            docnames = []
            for i in docsandapps:
                docnames.append(getdocname(i[0]))
            return render_template('patientlogin.html',mess=f'No user with such contact number.',docnames=docnames,docsandapps=docsandapps,docname_docid=docname_docid,l=l,l2=l2,patname=patname) 
    else:
        docsandapps,l = retalldocsandapps()
        docname_docid,l2 = ret_docname_docspec()
        docnames = []
        for i in docsandapps:
            docnames.append(getdocname(i[0]))
        return render_template('patientlogin.html',mess=f'Please select a date after today.',docnames=docnames,docsandapps=docsandapps,docname_docid=docname_docid,l=l,l2=l2,patname=patname) 


## Approve Doctor and add in registered doctors
@app.route('/approvedoctor')
def approvedoctor():
    doctoapprove = request.values['docid']
    c.execute('SELECT * FROM doctors')
    conn.commit()
    doctor_requests = c.fetchall()
    for i in doctor_requests:
        if str(i[5])==str(doctoapprove):
            c.execute(f"UPDATE doctors SET status=1 WHERE doc_id={str(doctoapprove)}")
            conn.commit()
            patient_reg_requests = ret_patient_reg_requests()
            doctor_reg_requests = ret_doctor_reg_requests()
            registered_patients = ret_registered_patients()
            registered_doctors = ret_registered_doctors()
            l1 = len(patient_reg_requests)
            l2 = len(doctor_reg_requests)
            l3 = len(registered_patients)
            l4 = len(registered_doctors)
            return render_template('adminlogin.html',mess=f'Doctor Approved successfully!!!',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 
    else:
        patient_reg_requests = ret_patient_reg_requests()
        doctor_reg_requests = ret_doctor_reg_requests()
        registered_patients = ret_registered_patients()
        registered_doctors = ret_registered_doctors()
        l1 = len(patient_reg_requests)
        l2 = len(doctor_reg_requests)
        l3 = len(registered_patients)
        l4 = len(registered_doctors)
        return render_template('adminlogin.html',mess=f'Doctor Not Approved...',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 


## Approve Patient and add in registered patients
@app.route('/approvepatient')
def approvepatient():
    pattoapprove = request.values['patnum']
    c.execute('SELECT * FROM patients')
    conn.commit()
    patient_requests = c.fetchall()
    for i in patient_requests:
        if str(i[3])==str(pattoapprove):
            c.execute(f"UPDATE patients SET status=1 WHERE phone_number={str(pattoapprove)}")
            conn.commit()
            patient_reg_requests = ret_patient_reg_requests()
            doctor_reg_requests = ret_doctor_reg_requests()
            registered_patients = ret_registered_patients()
            registered_doctors = ret_registered_doctors()
            l1 = len(patient_reg_requests)
            l2 = len(doctor_reg_requests)
            l3 = len(registered_patients)
            l4 = len(registered_doctors)
            return render_template('adminlogin.html',mess=f'Patient Approved successfully!!!',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 

    else:
        patient_reg_requests = ret_patient_reg_requests()
        doctor_reg_requests = ret_doctor_reg_requests()
        registered_patients = ret_registered_patients()
        registered_doctors = ret_registered_doctors()
        l1 = len(patient_reg_requests)
        l2 = len(doctor_reg_requests)
        l3 = len(registered_patients)
        l4 = len(registered_doctors)
        return render_template('adminlogin.html',mess=f'Patient Not Approved...',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 


## Approve an appointment request
@app.route('/doctorapproveappointment')
def doctorapproveappointment():
    docid = request.values['docid']
    patnum = request.values['patnum']
    patname = request.values['patname']
    appdate = request.values['appdate']
    c.execute(f"INSERT INTO doctorappointments VALUES ('{docid}','{patname}','{patnum}','{appdate}')")
    conn.commit()
    c.execute(f"DELETE FROM doctorappointmentrequests WHERE patientnum='{str(patnum)}'")
    conn.commit()
    appointment_requests_for_this_doctor,l1 = retapprequests(docid)
    fix_appointment_for_this_doctor,l2 = retdocsandapps(docid)
    return render_template('doctorlogin.html',appointment_requests_for_this_doctor=appointment_requests_for_this_doctor,fix_appointment_for_this_doctor=fix_appointment_for_this_doctor,l1=l1,l2=l2,docid=docid)


## Delete an appointment request
@app.route('/doctordeleteappointment')
def doctordeleteappointment():
    docid = request.values['docid']
    patnum = request.values['patnum']
    c.execute(f"DELETE FROM doctorappointmentrequests WHERE patientnum='{str(patnum)}'")
    conn.commit()
    appointment_requests_for_this_doctor,l1 = retapprequests(docid)
    fix_appointment_for_this_doctor,l2 = retdocsandapps(docid)
    return render_template('doctorlogin.html',appointment_requests_for_this_doctor=appointment_requests_for_this_doctor,fix_appointment_for_this_doctor=fix_appointment_for_this_doctor,l1=l1,l2=l2,docid=docid)


## Delete a doctor registration request
@app.route('/deletedoctorrequest')
def deletedoctorrequest():
    docid = request.values['docid']
    c.execute(f"DELETE FROM doctors WHERE doc_id='{str(docid)}'")
    conn.commit()
    patient_reg_requests = ret_patient_reg_requests()
    doctor_reg_requests = ret_doctor_reg_requests()
    registered_patients = ret_registered_patients()
    registered_doctors = ret_registered_doctors()
    l1 = len(patient_reg_requests)
    l2 = len(doctor_reg_requests)
    l3 = len(registered_patients)
    l4 = len(registered_doctors)
    return render_template('adminlogin.html',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 


## Delete a patient registration request
@app.route('/deletepatientrequest')
def deletepatientrequest():
    patnum = request.values['patnum']
    c.execute(f"DELETE FROM patients WHERE phone_number='{str(patnum)}'")
    conn.commit()
    patient_reg_requests = ret_patient_reg_requests()
    doctor_reg_requests = ret_doctor_reg_requests()
    registered_patients = ret_registered_patients()
    registered_doctors = ret_registered_doctors()
    l1 = len(patient_reg_requests)
    l2 = len(doctor_reg_requests)
    l3 = len(registered_patients)
    l4 = len(registered_doctors)
    return render_template('adminlogin.html',patient_reg_requests=patient_reg_requests,doctor_reg_requests=doctor_reg_requests,registered_patients=registered_patients,registered_doctors=registered_doctors,l1=l1,l2=l2,l3=l3,l4=l4) 


@app.route('/updatepatient')
def updatepatient():
    phn = request.args['phn']
    fn,ln,dob,phn,passw,add,status = getpatdetails(phn)
    return render_template('updatepatient.html',fn=fn,ln=ln,dob=dob,phn=phn,passw=passw,add=add,status=status) 


@app.route('/updatedoctor')
def updatedoctor():
    docid = request.args['docid']
    fn,ln,dob,phn,add,docid,passw,spec,status = getdocdetails(docid)
    return render_template('updatedoctor.html',fn=fn,ln=ln,dob=dob,phn=phn,passw=passw,add=add,status=status,spec=spec,docid=docid) 

@app.route('/makedoctorupdates',methods=['GET','POST'])
def makedoctorupdates():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    dob = request.form['dob']
    phn = request.form['phn']
    address = request.form['address']
    docid = request.args['docid']
    spec = request.form['speciality']
    c.execute("UPDATE doctors SET first_name=(?) WHERE doc_id=(?)",(firstname,docid))
    conn.commit()
    c.execute("UPDATE doctors SET last_name=(?) WHERE doc_id=(?)",(lastname,docid))
    conn.commit()
    c.execute("UPDATE doctors SET dob=(?) WHERE doc_id=(?)",(dob,docid))
    conn.commit()
    c.execute("UPDATE doctors SET phone_number=(?) WHERE doc_id=(?)",(phn,docid))
    conn.commit()
    c.execute("UPDATE doctors SET address=(?) WHERE doc_id=(?)",(address,docid))
    conn.commit()
    c.execute("UPDATE doctors SET speciality=(?) WHERE doc_id=(?)",(spec,docid))
    conn.commit()
    return render_template('home.html',mess='Updations Done Successfully!!!') 

    
@app.route('/makepatientupdates',methods=['GET','POST'])
def makepatientupdates():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    dob = request.form['dob']
    phn = request.args['phn']
    address = request.form['address']
    c.execute("UPDATE patients SET first_name=(?) WHERE phone_number=(?)",(firstname,phn))
    conn.commit()
    c.execute("UPDATE patients SET last_name=(?) WHERE phone_number=(?)",(lastname,phn))
    conn.commit()
    c.execute("UPDATE patients SET dob=(?) WHERE phone_number=(?)",(dob,phn))
    conn.commit()
    c.execute("UPDATE patients SET address=(?) WHERE phone_number=(?)",(address,phn))
    conn.commit()
    return render_template('home.html',mess='Updations Done Successfully!!!') 


###
# The functions below should be applicable to all Flask apps.
###

# Display Flask WTF errors as Flash messages
def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error in the %s field - %s" % (
                getattr(form, field).label.text,
                error
            ), 'danger')

@app.route('/<file_name>.txt')
def send_text_file(file_name):
    """Send your static text file."""
    file_dot_text = file_name + '.txt'
    return app.send_static_file(file_dot_text)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also tell the browser not to cache the rendered page. If we wanted
    to we could change max-age to 600 seconds which would be 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


@app.errorhandler(404)
def page_not_found(error):
    """Custom 404 page."""
    return render_template('404.html'), 404
