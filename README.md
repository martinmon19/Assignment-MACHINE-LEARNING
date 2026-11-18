# Hotel Booking Cancellation Prediction

- **Team:** _team name_
- **Course:** Machine Learning [P375], CUNEF Universidad
- **Repository:** https://github.com/martinmon19/Assignment-MACHINE-LEARNING

**Authors**
- Martín Monasterio (_cunef email_)
- Jorge Carro (_cunef email_)
- Javier González (_cunef email_)

---

## 1. Business problem

Hotels lose revenue when bookings are cancelled: a room that was blocked for weeks may end up empty,
and last-minute cancellations are hard to resell. In this dataset, about 37% of all bookings
(27.7% after removing duplicates) end in a cancellation or no-show.

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
- **Features:** booking information known at the time of booking: hotel, lead time, arrival date,
  length of stay, guests, meal, country, market segment, distribution channel, customer history,
  room type, deposit type, customer type, daily rate (`adr`), parking and special requests.
- **Excluded features (data leakage):** `reservation_status` matches the target exactly
  (*Canceled* and *No-Show* → 1, *Check-Out* → 0) and `reservation_status_date` is recorded after
  the outcome. Using them would give a perfect but useless model.
- **Class balance:** _to be completed with the result of `03_distributions`._

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
- Erroneous bookings removed: negative rate, 0-night stays, bookings with no guests and extreme guest counts.
- Final dataset: 86,622 bookings.

**Outliers (`02_outliers`)**
- Extreme values in `adr`, `lead_time` and total nights detected with the 1.5 × IQR rule, with
  limits computed **per hotel** because their distributions differ (e.g. 7-night stays are 17% of
  resort bookings but 3.5% of city bookings).
- 6,442 bookings (7.4%) flagged. After review, only one is excluded (a rate of 5,400, a data entry
  error); the rest are rare but plausible (groups booking far in advance, long holiday stays, free stays).
- Flagged bookings are cancelled more often (40.5% vs 26.7%), especially long stays and long lead
  times, so extreme values carry useful information for the model.

**Distributions (`03_distributions`)**
- _To be completed._
