# Hotel Booking Cancellation Prediction

- **Course:** Machine Learning [P375], CUNEF Universidad
- **Repository:** https://github.com/martinmon19/Assignment-MACHINE-LEARNING

**Authors**
- Martín Monasterio (martin.monasterio@cunef.edu)
- Javier González (j.gonzalezdimas@cunef.edu)
- Jorge Carro (jorge.carro@cunef.edu)

---

## 1. Business problem

Hotels lose revenue when bookings are cancelled: a room that was blocked for weeks may end up empty,
and last-minute cancellations are hard to resell. In this dataset, about 37% of all bookings
(27.7% after cleaning the data) end in a cancellation or no-show.

Hotels usually react with general policies:
- **Overbooking** based on an average cancellation rate. If the estimate is wrong, the hotel either
  has empty rooms or has to relocate guests, with compensation costs and reputational damage.
- **Non-refundable rates and deposits** applied to every customer. They reduce cancellations but
  also discourage bookings from customers who would not have cancelled.

These approaches treat all bookings the same. Predicting the **probability that each individual
booking will be cancelled** would allow targeted actions: adjusting overbooking with real risk,
asking for a deposit only on high-risk bookings, or contacting those customers in advance.

## 2. Data

**Source:** Hotel booking demand dataset (`hotels.csv`), published in TidyTuesday:
https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11

**Original article:** Antonio, N., de Almeida, A. and Nunes, L. (2019). *Hotel booking demand datasets*.
Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

The dataset contains 119,390 bookings and 32 columns from two hotels in Portugal (a city hotel and
a resort hotel) with arrivals between July 2015 and August 2017.

| Advantages | Disadvantages |
|---|---|
| Real data from hotel management systems | Only two hotels, both in Portugal: results may not generalise |
| Large sample (119,390 bookings) | Data from 2015–2017, before COVID-19 changed travel behaviour |
| Many variables: dates, stay, guests, channel, price, customer history | No booking ID: exact duplicates cannot be told apart from identical real bookings |
| Public and documented in a peer-reviewed article | Agents and companies are anonymised as numeric codes |
| | Some columns are only known after the outcome (data leakage, see below) |

**How to get the data:** download `hotels.csv` from the link above and place it in `data/`.
The `data/` folder is empty in this repository on purpose.

## 3. Problem formulation

- **Type of problem:** supervised learning, **binary classification**.
- **Target:** `is_canceled` (1 = cancelled or no-show, 0 = checked out).
- **Features:** 23 columns known at the time of booking: `hotel`, `lead_time`, `arrival_date_month`,
  `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`, `adults`,
  `children`, `babies`, `meal`, `country`, `market_segment`, `distribution_channel`,
  `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled`,
  `reserved_room_type`, `deposit_type`, `agent`, `company`, `customer_type`, `adr` and
  `total_of_special_requests`. Each booking $i$ is described by a feature vector
  $\mathbf{x}_i = (x_{i1}, \dots, x_{ip})$ with $p = 23$, and the model estimates
  $P(y = 1 \mid \mathbf{x})$.
- **Excluded features:**
  - *Leakage, outcome:* `reservation_status` matches the target exactly (*Canceled* and *No-Show* → 1,
    *Check-Out* → 0) and `reservation_status_date` is the date of that final status. Using them would
    give a perfect but useless model.
  - *Leakage, recorded after the booking:* `assigned_room_type` (decided by the hotel, often at
    check-in), `required_car_parking_spaces` (none of the 7,290 bookings that asked for parking was
    cancelled), `booking_changes` (counted until check-in or cancellation) and `days_in_waiting_list`
    (fixed once the booking is confirmed).
  - *Other reasons:* `arrival_date_year` (a model is applied to future years) and
    `arrival_date_week_number` (repeats the month).
- **Class balance:** 27.7% of the 86,621 clean bookings are cancelled and 72.3% are not. Always
  predicting "not cancelled" already gives 72.3% accuracy, so models will be evaluated with recall,
  precision and AUC as well as accuracy, using a stratified train/test split.

## 4. Project structure

```
README.md
data/                          ← empty; place hotels.csv here
notebooks/
    01_cleaning.ipynb          ← data types, missing values, duplicates, erroneous values
    02_outliers.ipynb          ← detection and treatment of extreme values
    03_distributions.ipynb     ← distributions and relationship with cancellation
```

**How to run:** run the notebooks in order. `01_cleaning` creates `data/hotels_clean.csv`,
which is used by the other two notebooks.

## 5. Key findings and actions

**Data cleaning (`01_cleaning`)**
- 31,994 exact duplicates (26.8%) removed: without a booking ID they could repeat the same booking
  in training and test sets. The cancellation rate drops from 37.0% to 27.5%.
- Missing values filled: `children` (0), `country` (*Unknown*), `agent` (*No agent*), `company` (*No company*).
- Erroneous bookings removed: negative rate, one rate of 5,400 (a data entry error), 0-night stays, bookings with no guests and extreme guest counts.
- Final dataset: 86,621 bookings.

**Outliers (`02_outliers`)**
- Extreme values in `adr`, `lead_time` and total nights detected with the 1.5 × IQR rule, with
  limits computed **per hotel** because their distributions differ (e.g. 7-night stays are 17% of
  resort bookings but 3.5% of city bookings).
- 6,441 bookings (7.4%) flagged. They are rare but plausible (groups booking far in advance, long holiday stays, free stays) and are kept; the only clear error, a rate of 5,400, was already removed in `01_cleaning`.
- Flagged bookings are cancelled more often (40.5% vs 26.7%), especially long stays and long lead
  times, so extreme values carry useful information for the model.

**Distributions (`03_distributions`)**
- Cancellation rate rises with **lead time** (8.5% for bookings made a week or less ahead vs about 40%
  for more than six months) and with **price** (15.6% below 50 per night vs 37.6% above 200).
- **Customer history** separates the classes strongly: customers who cancelled before cancel again
  68.2% of the time, while repeated guests (8.2%) and customers with previous stays (5.0%) rarely cancel.
- Individual online bookings cancel the most (*Online TA* 35.5%); bookings with an agreement cancel much
  less (*Corporate* 12.2%, *Group* 7.6%). The City Hotel cancels more than the Resort Hotel (30.2% vs 23.7%),
  and cancellations peak in summer.
- *Non Refund* deposits are cancelled 94.7% of the time: they are mostly group and agency blocks, not
  individual customers.
- **Leakage check:** besides `reservation_status` and its date, four columns are recorded or updated
  after the booking is made and are excluded: `assigned_room_type`, `required_car_parking_spaces`
  (0% cancellations), `booking_changes` and `days_in_waiting_list`. 23 candidate features remain.
- Actions: mostly-zero counts turned into yes/no variables, rare categories grouped into "Other"
  (top 10 countries and agents), "Undefined" merged into "SC" in `meal`.
