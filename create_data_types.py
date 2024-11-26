import json
## Create .json of data types 

# define target
target = 'fraud_bool'

nominal = ['payment_type',
 'employment_status',
 'housing_status',
 'source',
 'device_os',
 'income',
 'customer_age',
 'email_is_free',
 'phone_home_valid',
 'phone_mobile_valid',
 'has_other_cards',
 'foreign_request',
 'keep_alive_session']

ordinal = ['income', 
 'customer_age', 
 'month']

numerical = ['name_email_similarity',
 'prev_address_months_count',
 'current_address_months_count',
 'days_since_request',
 'intended_balcon_amount',
 'zip_count_4w',
 'velocity_6h',
 'velocity_24h',
 'velocity_4w',
 'bank_branch_count_8w',
 'date_of_birth_distinct_emails_4w',
 'credit_risk_score',
 'bank_months_count',
 'proposed_credit_limit',
 'session_length_in_minutes',
 'device_fraud_count',
 'device_distinct_emails_8w',
]

# create dic
data_types = {}
data_types['target'] = target
data_types['nominal'] = nominal
data_types['ordinal'] = ordinal
data_types['numerical'] = numerical

# write json file
with open("data_types.json", "w") as json_file:
    json.dump(data_types, json_file, indent=4)

print("datatypes.json has been created successfully.")