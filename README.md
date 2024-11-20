# Kaggle Dataset
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data

Feature:
- income - annual income grouped into quantiles (between 0 and 1)
- name_email_similarity - how similar is email to applicant name (scored between 0 and 1)
- prev_address_months_count (-1 is missing value)
- current_address_months_count (-1 is missing value)
- customer_age - age in bins of decades (20-29 is recorded as 20)
- days_since_request
- intended_balcon_amount - Initial transferred amount for application
- payment_type - annonymised values between {'AA', 'AB', 'AC', 'AD', 'AE'}
- zip_count_4w - Number of applications within same zip code in last 4 weeks
- velocity_6h - Velocity of total applications made in last 6 hours i.e., average - number of applications per hour in the last 6 hours
- velocity_24h - average number of applications per hour in the last 24 hours
v- elocity_4w - average number of applications per hour in the last 4 weeks
- bank_branch_count_8w - Number of total applications in the selected bank branch in last 8 weeks.
- date_of_birth_distinct_emails_4w - Number of emails for applicants with same date of birth in last 4 weeks.
- employment_status = annonymised values between {'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'}
- credit_risk_score - internal score of application risk
- emaail_is_free - domain of application email is free or paid for
- housing_status - annonymised values between {'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'}
- phone_home_valid 
- phone_mobile_valid
- bank_months_count - for how many months was previous account held (-1 is missing value)
- has_other_cards - applicant has other cards from the same banking company
- proposed_credit_limit - applicants proposed credit limit should be between 200 and 2000
- foreign_request - rigin country of request is different from banks's country
- source - brower or app (INTERNET or TELEAPP)
- session_length_in_minutes - length of user session in minutes
- device_os - operative system of device
- keep_alive_session - user option on session logout
- device_distinct_emails_8w - Number of distinct emails in banking website from the used device in last 8 weeks. 
- device_fraud_count - Number of fraudulent applications with used device.
- month - month where ther application was made (0 to 7) ??