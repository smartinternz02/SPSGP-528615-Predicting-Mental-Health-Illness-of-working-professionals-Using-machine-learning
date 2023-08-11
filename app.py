from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

# Initialize the LabelEncoders for categorical columns
label_encoders = {
    'Gender': LabelEncoder(),
    'self_employed': LabelEncoder(),
    'family_history': LabelEncoder(),
    'work_interfere': LabelEncoder(),
    'no_employees': LabelEncoder(),
    'remote_work': LabelEncoder(),
    'tech_company': LabelEncoder(),
    'benefits': LabelEncoder(),
    'care_options': LabelEncoder(),
    'wellness_program': LabelEncoder(),
    'seek_help': LabelEncoder(),
    'anonymity': LabelEncoder(),
    'Leave': LabelEncoder(),
    'mental_health_consequence': LabelEncoder(),
    'phys_health_consequence': LabelEncoder(),
    'coworkers': LabelEncoder(),
    'supervisor': LabelEncoder(),
    'mental_health_interview': LabelEncoder(),
    'phys_health_interview': LabelEncoder(),
    'mental_vs_physical': LabelEncoder(),
    'obs_consequence': LabelEncoder()
}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pred') 
def predict():
    return render_template("index.html")

@app.route('/out', methods=["POST"])
def output():
    # Define the data dictionary with form values
    data = {
        'Age': request.form["Age"],
        'Gender': request.form["Gender"],
        'self_employed': request.form["self_employed"],
        'family_history': request.form["family_history"],
        'work_interfere': request.form["work_interfere"],
        'no_employees': request.form["no_employees"],
        'remote_work': request.form["remote_work"],
        'tech_company': request.form["tech_company"],
        'benefits': request.form["benefits"],
        'care_options': request.form["care_options"],
        'wellness_program': request.form["wellness_program"],
        'seek_help': request.form["seek_help"],
        'anonymity': request.form["anonymity"],
        'Leave': request.form["Leave"],
        'mental_health_consequence': request.form["mental_health_consequence"],
        'phys_health_consequence': request.form["phys_health_consequence"],
        'coworkers': request.form["coworkers"],
        'supervisor': request.form["supervisor"],
        'mental_health_interview': request.form["mental_health_interview"],
        'phys_health_interview': request.form["phys_health_interview"],
        'mental_vs_physical': request.form["mental_vs_physical"],
        'obs_consequence': request.form["obs_consequence"]
    }

    # Fit the LabelEncoders to the data
    for column, encoder in label_encoders.items():
        encoder.fit([data[column]])

    # Convert categorical features to numerical using label encoding
    encoded_data = {
        'Age': int(data['Age']),
        'Gender': label_encoders['Gender'].transform([data['Gender']])[0],
        'self_employed': label_encoders['self_employed'].transform([data['self_employed']])[0],
        'family_history': label_encoders['family_history'].transform([data['family_history']])[0],
        'work_interfere': label_encoders['work_interfere'].transform([data['work_interfere']])[0],
        'no_employees': label_encoders['no_employees'].transform([data['no_employees']])[0],
        'remote_work': label_encoders['remote_work'].transform([data['remote_work']])[0],
        'tech_company': label_encoders['tech_company'].transform([data['tech_company']])[0],
        'benefits': label_encoders['benefits'].transform([data['benefits']])[0],
        'care_options': label_encoders['care_options'].transform([data['care_options']])[0],
        'wellness_program': label_encoders['wellness_program'].transform([data['wellness_program']])[0],
        'seek_help': label_encoders['seek_help'].transform([data['seek_help']])[0],
        'anonymity': label_encoders['anonymity'].transform([data['anonymity']])[0],
        'Leave': label_encoders['Leave'].transform([data['Leave']])[0],
        'mental_health_consequence': label_encoders['mental_health_consequence'].transform([data['mental_health_consequence']])[0],
        'phys_health_consequence': label_encoders['phys_health_consequence'].transform([data['phys_health_consequence']])[0],
        'coworkers': label_encoders['coworkers'].transform([data['coworkers']])[0],
        'supervisor': label_encoders['supervisor'].transform([data['supervisor']])[0],
        'mental_health_interview': label_encoders['mental_health_interview'].transform([data['mental_health_interview']])[0],
        'phys_health_interview': label_encoders['phys_health_interview'].transform([data['phys_health_interview']])[0],
        'mental_vs_physical': label_encoders['mental_vs_physical'].transform([data['mental_vs_physical']])[0],
        'obs_consequence': label_encoders['obs_consequence'].transform([data['obs_consequence']])[0]
    }

    pred = model.predict([list(encoded_data.values())])
    pred = pred[0]

    if pred:
        return render_template("output.html", y="This person requires mental health treatment")
    else:
        return render_template("output.html", y="This person doesn't require mental health treatment")

if __name__ == '__main__':
    app.run(debug=True)
