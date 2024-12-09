import json
## Create .json of data types 

# define target
target = 'fraud_bool'

nominal = ['payment_type',
 'employment_status',
 'housing_status',
 'source',
 'device_os',
 'email_is_free',
 'phone_home_valid',
 'phone_mobile_valid',
 'has_other_cards',
 'foreign_request',
 'keep_alive_session',
 'proposed_credit_limit' # added here because of aparent bucketing
]

nominal_multi_category = ['payment_type',
 'employment_status',
 'housing_status',
 'device_os',
 'proposed_credit_limit' # added here because of aparent bucketing
]

nominal_binary = ['source_is_internet_not_teleapp',
 'email_is_free',
 'phone_home_valid',
 'phone_mobile_valid',
 'has_other_cards',
 'foreign_request',
 'keep_alive_session']

ordinal = ['income', 
 'customer_age']

temporal = ['month']

numerical_discrete = ['prev_address_months_count',
 'current_address_months_count', 
 'bank_branch_count_8w',
 'date_of_birth_distinct_emails_4w',
 'credit_risk_score',
 'bank_months_count',
 'device_fraud_count',
 'device_distinct_emails_8w',
 'zip_count_4w'
]
numerical_continuous_bounded = ['name_email_similarity']

numerical_continuous_unbounded = ['days_since_request',
 'intended_balcon_amount',
 'velocity_6h',
 'velocity_24h',
 'velocity_4w',
 'session_length_in_minutes'
 ]

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
 'session_length_in_minutes',
 # 'device_fraud_count',  # removed because always 0
 'device_distinct_emails_8w',
]

# create dic
data_types = {}
data_types['target'] = target
data_types['nominal'] = nominal
data_types['ordinal'] = ordinal
data_types['numerical'] = numerical
data_types['temporal'] = temporal



data_types['numerical_discrete'] = numerical_discrete
data_types['numerical_continuous_bounded'] = numerical_continuous_bounded
data_types['numerical_continuous_unbounded'] = numerical_continuous_unbounded
data_types['nominal_multi_category'] = nominal_multi_category
data_types['nominal_binary'] = nominal_binary


# write json file
with open("data_types.json", "w") as json_file:
    json.dump(data_types, json_file, indent=4)

print("datatypes.json has been created successfully.")