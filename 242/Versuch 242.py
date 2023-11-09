#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import os
import tabulate
import latex_table as tab
from kafe2 import XYContainer, Fit, Plot, ContoursProfiler
from math import log10 , floor
import math
import pandas as pd
from colorama import Fore, Back, Style


# In[2]:


# Directory where the data is located
directory = "../V242/"  # Directory

# Set and display the directory
os.chdir(directory)  # Navigate to the desired directory
directory_path = os.getcwd()  # Current directory
print("My current directory is: " + directory_path)
folder_name = os.path.basename(directory_path)  # Current folder
print("My directory name is: " + folder_name)


# In[3]:


######################################## Define Functions ########################################

#################### Print Values ####################
Print_Value = True
def print_value(Variable, Unit, Value):
    if Print_Value:
        print(Variable, "in", Unit + ":", Value)
# Example usage:
# a = 1.234
# print_value("a", "m", a)

#################### Print Errors ####################
Print_Error = True
def print_error(Variable, Unit, ErrorValue):
    if Print_Error:
        print("Error on", Variable, "in", Unit + ":", ErrorValue)
# Example usage:
# a_err = 0.5678
# print_error("a", "m", a_err)

#################### Insert an item between elements in a list ####################
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

#################### Round Values to 3 Significant Figures ####################
def round_to_3sigfigs(x, sig):
    if x == 0:  # Since the round function doesn't work with 0
        return 0
    elif x > 0:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)
    elif x < 0:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)
    elif math.isnan(x):
        return x
    else:
        return "error"
# Example usage:
# a = 1.234
# print(round_to_3sigfigs(a, 3))
# a2 = -1.234
# print(round_to_3sigfigs(a2, 3))
# a3 = 0.0000
# print(round_to_3sigfigs(a3, 3)

#################### Round Elements in a List to 3 Significant Figures ####################
def round_list(lst, k):
    list_rounded = []
    for element in lst:
        if isinstance(element, (float, np.float64)):
            a = round_to_3sigfigs(element, k)
            list_rounded.append(a)
        else:
            list_rounded.append(element)
    return list_rounded
# Example usage:
# a1 = [1.234, 5.678]
# print(round_list_elements(a1, 3))


# In[4]:


# 242_b
print(f'{"242.b)":#^50}')

DataFile = "Versuch242.txt"  # File name

data = np.loadtxt(DataFile)
print(data)

# Extract columns from the file
column1 = data[:, 0]  # 1st column
column2 = data[:, 1]  # 2nd column
column3 = data[:, 2]  # 3rd column
column4 = data[:, 3]  # 4th column
column5 = data[:, 4]  # 5th column
column6 = data[:, 5]  # 6th column

# Define N as the number of measurement pairs
N = len(column1)
print("N:", N)


# In[5]:


#################### Literature Values ####################
print(f'{"Literature Values":#^50}')

Mu_0 = 1.25663706212e-6  # in N/A^-2, Literature value https://physics.nist.gov/cgi-bin/cuu/Value?mu0 (As of 02.11.2022)

Mu_0_err = 1.9e-16  # in N/A^-2, Literature value https://physics.nist.gov/cgi-bin/cuu/Value?mu0 (As of 02.11.2022)
print("Error on Mu_0:", Mu_0_err)

#################### Measured Values ####################
print("\n", f'{"Measured Values":#^50}')

Measurement = column1  # Integer

Voltage = column2  # in V
Distance = column5-column6  # in m
Current_1 = column3  # in A
Current_2 = column4  # in A

Number_of_Windings = 130  # Coil winding number
Coil_Radius = 0.15  # in m, coil radius

#################### Measured Errors ####################
print("\n", f'{"Measured Errors":#^50}')

Measurement_err = np.array([0] * N)  # 0, as per the count of measurements
print("Error on measurement number:", Measurement_err)

Voltage_err = np.array([1] * N)  # in V
print("Error on Voltage in V:", Voltage_err)
Distance_err = np.array([0.005] * N)  # in m
print("Error on Distance in m:", Distance_err)
Current_1_err = np.array([0.01] * N)  # in A
print("Error on Current_1 in A:", Current_1_err)
Current_2_err = np.array([0.01] * N)  # in A
print("Error on Current_2 in A:", Current_2_err)

Number_of_Windings_err = 0  # 0, as per the count
Coil_Radius_err = 0  # in m, 0 as given in the manual

#################### Calculate Values ####################
print("\n", f'{"Calculate Values":#^50}')

Radius = Distance / 2
print("Radius r in m:", Radius)
Average_Current = (Current_1 + Current_2) / 2
print("Current I in A:", Average_Current)

# Calculate B_S (Magnetic Flux Density of the Coil)
B_S = (4/5) ** (3/2) * Mu_0 * ((Number_of_Windings * Average_Current) / Coil_Radius)
print("Magnetic Flux Density of the Coil B_s in T:", B_S)

# Calculate values for the graph
x_values = Voltage
print("x-Values Voltage U:", x_values)
y_values = (Radius * Average_Current) ** 2
print("y-Values (r * I)^2:", y_values)

#################### Calculate Errors ####################
print("\n", f'{"Calculate Errors":#^50}')

s1 = (1 / 2) * Distance_err
Radius_err = (s1 ** 2) ** (1 / 2)
print("Error on Radius in m:", Radius_err)

s1 = (Current_2 / 2) * Current_1_err
s2 = (Current_1 / 2) * Current_2_err
Current_err = (s1 ** 2 + s2 ** 2) ** (1 / 2)
print("Error on Current I in A:", Current_err)

s1 = (((4/5) ** (3/2) * Mu_0 * Number_of_Windings / Coil_Radius)) * Current_err
B_S_err = (s1 ** 2) ** (1 / 2)
print("Error on B_s in T:", B_S_err)

x_errors = Voltage_err
print("Error on x-Values:", x_errors)

s1 = (2 * Radius * Average_Current ** 2) * Radius_err
s2 = (2 * Radius ** 2 * Average_Current) * Current_err
y_errors = (s1 ** 2 + s2 ** 2) ** (1 / 2)
print("Error on y-Values", y_errors)

x_data = x_values
y_data = y_values
x_err = x_errors
y_err = y_errors


# In[6]:


#################### Create LaTeX Table ####################

# print("Type of data:", type(data))
# print("Dimension of data:", data.ndim)

k = 3
column1_rounded = round_list(x_values, k)
column2_rounded = round_list(x_errors, k)
column3_rounded = round_list(y_values, k)
column4_rounded = round_list(y_errors, k)

table1 = tab.LatexTable([column1_rounded, column2_rounded, column3_rounded, column4_rounded])
print(table1)

print("\n")

print("$U$", "&", "$\Delta U$", "&", "$r^2I^2$", "&", "$\Delta r^2I^2$")


# In[7]:


#################### Calculate Linear Fit and Plot ####################

# Print data for reference
print("x:", x_data)
print("x_err:", x_err)
print("y:", y_data)
print("y_err:", y_err)

# N = Number of data pairs
print("Number of data pairs N:", N)

# Rename variables for clarity
x_values = x_data
y_values = y_data
# Error x_err and y_err are carried over from the previous cell

# Calculate variance-weighted means z ∈ {x, y, xy, x^2}

# Calculate the numerator and denominator for z/y_err**2
sum_y_err = sum(1 / (y_err) ** 2)
x_div_y_err = sum(x_values / (y_err) ** 2)
y_div_y_err = sum(y_values / (y_err) ** 2)
x2_div_y_err = sum((x_values ** 2) / (y_err) ** 2)
xy_div_y_err = sum((x_values * y_values) / (y_err) ** 2)

# Calculate variance-weighted means
x_mean = x_div_y_err / sum_y_err
y_mean = y_div_y_err / sum_y_err
x2_mean = x2_div_y_err / sum_y_err
xy_mean = xy_div_y_err / sum_y_err

# Calculate m and n:
m_fit = (xy_mean - (x_mean * y_mean)) / (x2_mean - (x_mean) ** 2)  # Slope m
print("Slope m: ", m_fit)
n_fit = y_mean - m_fit * x_mean  # y-intercept n
print("y-intercept n: ", n_fit)

# Calculate variances:
sigma_y_mean = N / sum_y_err  # Variance-weighted standard deviation Sigmay
Vm = sigma_y_mean / (16 * (x2_mean - (x_mean) ** 2))  # Variance of the slope
print("Variance of m (V[m]): ", Vm)
Vn = x2_mean * Vm  # Variance of the y-intercept
print("Variance of n (V[n]): ", Vn)
Vmn = -Vm * x_mean  # Covariance of m and n
print("Covariance of m and n (Vmn): ", Vmn)

# Calculate Goodness of Fit (Güte):
xi2_array = (y_values - m_fit * x_values - n_fit) ** 2 / y_err
xi2 = sum(xi2_array)
gute = xi2 / (N - 2)  # Goodness of Fit
print("Goodness of Fit (Güte): ", gute)


def linear_f(x, m, b):  # Define the linear function
    return m * x + b


def lin_fit(x_data, y_data, x_err, y_err):  # Create a function with data as arguments
    xy_data = XYContainer(x_data, y_data)  # Create a container for x and y
    xy_data.label = "Measured Data"
    xy_data.add_error(axis='x', err_val=x_err)  # x-error bars
    xy_data.add_error(axis='y', err_val=y_err)  # y-error bars
    linear_fit = Fit(data=xy_data, model_function=linear_f)  # Create an object with the xy container and the linear function
    fit_results = linear_fit.do_fit()  # Calculate and display results later
    plot = Plot(fit_objects=linear_fit)
    plot.x_label = "$U$"  # x-axis label
    plot.y_label = "$r^2I^2$"  # y-axis label
    plot.plot()
    plt.savefig('242b_Fit.jpg', dpi=300)  # Save the plot as a jpg and display it
    plt.show()  # Show the plot
    return fit_results

# Display results in the covariance matrix
print(lin_fit(x_data, y_data, x_err, y_err)['parameter_cov_mat'])


# In[8]:


#################### Calculate e/m from the Formula ####################

# Calculate from the formula
specific_charge = (2 * Voltage) / (Radius ** 2 * B_S ** 2)
print("e/m:", specific_charge)

# Calculate the average of e/m from the formula
specific_charge_avg = np.average(specific_charge)
print("Average of e/m from the formula:", specific_charge_avg)

#################### Calculate Errors ####################

s1 = (2 / (Radius ** 2 * B_S ** 2)) * Voltage_err
s2 = ((-4 * Voltage) / (Radius ** 3 * B_S ** 2)) * Radius_err
s3 = ((-4 * Voltage) / (Radius ** 2 * B_S ** 3)) * B_S_err
specific_charge_err = (s1 ** 2 + s2 ** 2 + s3 ** 2) ** (1 / 2)
print("Error on e/m from the formula:", specific_charge_err)

specific_charge_err_avg = np.average(specific_charge_err)
print("Average error on e/m from the formula:", specific_charge_err_avg)

#################### Calculate e/m from the Fit ####################

specific_charge_fit = (1 / m_fit) * ((2 * Coil_Radius ** 2) / (((4 / 5) ** (3/2)) ** 2 * Mu_0 ** 2 * Number_of_Windings ** 2))
print("e/m from the fit in C/kg:", specific_charge_fit)

#################### Calculate Errors ####################

s1 = ((-1 / m_fit ** 2) * ((2 * Coil_Radius ** 2) / (((4 / 5) ** (3/2)) ** 2 * Mu_0 ** 2 * Number_of_Windings ** 2))) * Vm
specific_charge_fit_err = (s1 ** 2) ** (1 / 2)
print("Error on e/m from the fit:", specific_charge_fit_err)

#################### Literature Value ####################

specific_charge_lit = -1.75882001076e11  # in C/kg, https://physics.nist.gov/cgi-bin/cuu/Value?esme
print("Literature value for e/m:", specific_charge_lit)
specific_charge_lit_err = 0.00000000053e11  # in C/kg,  https://physics.nist.gov/cgi-bin/cuu/Value?esme
print("Error on literature value for e/m:", specific_charge_lit_err)


# In[9]:


#################### Calculate B_E (Earth's Magnetic Field) ####################

I_diff = abs(Current_1 - Current_2) / 2
print("I_diff in A:", I_diff)

B_E = (4/5) ** (3/2) * Mu_0 * (Number_of_Windings / Coil_Radius) * I_diff
print("Earth's Magnetic Field B_E in T:", B_E)

B_E_avg = np.average(B_E)
print("Average of B_E in T:", B_E_avg)

#################### Calculate Errors ####################

s1 = (1 / 2) * Current_1_err
s2 = (1 / 2) * Current_2_err
I_diff_err = (s1 ** 2 + s2 ** 2) ** (1 / 2)
print("Error on I_diff in A:", I_diff_err)

s1 = ((4/5) ** (3/2) * Mu_0 * (Number_of_Windings / Coil_Radius)) * I_diff_err
B_E_err = (s1 ** 2) ** (1 / 2)
print("Error on B_E in T:", B_E_err)
print("Number of windings (n):", Number_of_Windings)
print("Coil radius (R):", Coil_Radius)

B_E_err_avg = np.average(B_E)
print("Average of B_E in T:", B_E_err_avg)

B_E_err_avg = np.average(B_E_err)
print("Average error on B_E in T:", B_E_err_avg)

#################### Literature Value ####################

B_E_lit = 20 / 1000 / 1000  # Conversion from µT to T (horizontal component) Source: https://www.chemie.de/lexikon/Erdmagnetfeld.html (As of 11/25/2022 10:19 AM)
print("Literature value of Earth's Magnetic Field in T:", B_E_lit)


# In[10]:


#################### Create LaTeX Table ####################

# print("Type of data:", type(data))
# print("Dimension of data:", data.ndim)

k = 3
B_E_column1_rounded = round_list(B_E, k)
B_E_column2_rounded = round_list(B_E_err, k)

table1 = tab.LatexTable([B_E_column1_rounded, B_E_column2_rounded])
print(table1)

print("\n")

print("$B_E$", "&", "$\Delta B_E$")


# In[11]:


#Lade die jeweiligen Tröpfchen in numpy arrays

T1 = np.loadtxt('./droplets/droplet_1.txt')
T2 = np.loadtxt('./droplets/droplet_2.txt')
T3 = np.loadtxt('./droplets/droplet_3.txt')
T4 = np.loadtxt('./droplets/droplet_4.txt')
T5 = np.loadtxt('./droplets/droplet_5.txt')
T6 = np.loadtxt('./droplets/droplet_6.txt')
T7 = np.loadtxt('./droplets/droplet_7.txt')
T8 = np.loadtxt('./droplets/droplet_8.txt')
T9 = np.loadtxt('./droplets/droplet_9.txt')
T10 = np.loadtxt('./droplets/droplet_10.txt')
T11 = np.loadtxt('./droplets/droplet_11.txt')
T12 = np.loadtxt('./droplets/droplet_12.txt')
T13 = np.loadtxt('./droplets/droplet_13.txt')
T14 = np.loadtxt('./droplets/droplet_14.txt')
T15 = np.loadtxt('./droplets/droplet_15.txt')
T16 = np.loadtxt('./droplets/droplet_16.txt')
T17 = np.loadtxt('./droplets/droplet_17.txt')
T18 = np.loadtxt('./droplets/droplet_18.txt')
T19 = np.loadtxt('./droplets/droplet_19.txt')
T20 = np.loadtxt('./droplets/droplet_20.txt')

#T4


# In[45]:


#Angaben zu Messinstrumenten etc.

s= 0.0005
ds = 0.00005
dt = 0.5
eta = 18.19*10**(-6)
deta = 0.06*10**(-6)
d_Platte = 0.00766
U = 300.0
dU = 10.0
E = U/d_Platte
dE = np.sqrt(U**2/d_Platte)
oel = 886.0
luft = 1.225
g = 9.81

Angaben_df = pd.DataFrame(np.array([s, ds, dt, eta, deta, d_Platte, U, dU, E, dE, oel, luft, g]))
Angaben_df.insert(0, '###', ['s', 'ds', 'dt', 'eta', 'deta', 'd_Platte', 'U', 'dU', 'E', 'dE', 'oel', 'luft', 'g'])

#Variabeln werden geprintet
print(f'{s = }', type(s))
print(f'{ds = }',type(ds))
print(f'{dt = }',type(dt))
print(f'{eta = }',type(eta))
print(f'{deta = }',type(deta))
print(f'{d_Platte = }',type(d_Platte))
print(f'{U = }',type(U))
print(f'{dU = }',type(dU))
print(f'{E = }',type(E))
print(f'{dE = }',type(dE))
print(f'{oel = }',type(oel))
print(f'{luft = }',type(luft))
print(f'{g = }',type(g))


# In[26]:


# Define a list to store DataFrames
dataframes = []

for i in range(1, 21):  # Loop from 1 to 20
    T_i = globals().get(f'T{i}')
    if T_i is not None:
        # Define the variables specific to each snippet
        droplet_name = f'Tropfen {i}'
        T_name = f'T{i}'

        # Calculate v and dv for each T column
        T_u = np.array((T_i)[:, 0])
        T_d = np.array((T_i)[:, 1])
        T_o = np.array((T_i)[:, 2])
        
        # Create masked arrays
        T_u_masked = ma.masked_equal(T_u, 0)
        T_d_masked = ma.masked_equal(T_d, 0)
        T_o_masked = ma.masked_equal(T_o, 0)

        # Calculate v_T_u, v_T_d, v_T_o
        v_T_u = s / T_u_masked
        v_T_d = s / T_d_masked
        v_T_o = s / T_o_masked

        # Calculate dv_T_u, dv_T_d, dv_T_o
        dv_T_u = np.sqrt((ds / T_u_masked) ** 2 + ((s * dt) / (T_u_masked ** 2)) ** 2)
        dv_T_d = np.sqrt((ds / T_d_masked) ** 2 + ((s * dt) / (T_d_masked ** 2)) ** 2)
        dv_T_o = np.sqrt((ds / T_o_masked) ** 2 + ((s * dt) / (T_o_masked ** 2)) ** 2)
        
        # calculate the means
        v_T_o_mean = np.mean(v_T_o)
        v_T_u_mean = np.mean(v_T_u)
        v_T_d_mean = np.mean(v_T_d)
        dv_T_o_mean = np.mean(dv_T_o)
        dv_T_u_mean = np.mean(dv_T_u)
        dv_T_d_mean = np.mean(dv_T_d)

        # Create a DataFrame for each droplet
        colName = '###'
        colvTu = f'Tropfen {i} up'
        coldvTu = f'+- up'
        colvTd = f'Tropfen {i} down'
        coldvTd = f'+- down'
        colvTo = f'Tropfen {i} off'
        coldvTo = f'+- off'

        df = pd.DataFrame({
            colName: [droplet_name, "NaN", "NaN", "NaN", "NaN"],
            colvTu: v_T_u,
            coldvTu: dv_T_u,
            colvTd: v_T_d,
            coldvTd: dv_T_d,
            colvTo: v_T_o,
            coldvTo: dv_T_o
        })

        # Append the mean row to the DataFrame
        mean_row = {
            colName: 'Mean',
            colvTu: v_T_u_mean,
            coldvTu: dv_T_u_mean,
            colvTd: v_T_d_mean,
            coldvTd: dv_T_d_mean,
            colvTo: v_T_o_mean,
            coldvTo: dv_T_o_mean
        }
        df = df.append(mean_row, ignore_index=True)

        dataframes.append(df)

# Access the DataFrames for specific droplets
# Adjust index for 0-based indexing
df_1 = dataframes[2 - 2]
df_2 = dataframes[3 - 2]
df_3 = dataframes[4 - 2]
df_4 = dataframes[5 - 2]
df_5 = dataframes[6 - 2]
df_6 = dataframes[7 - 2]
df_7 = dataframes[8 - 2]
df_8 = dataframes[9 - 2]
df_9 = dataframes[10 - 2]
df_10 = dataframes[11 - 2]
df_11 = dataframes[12 - 2]
df_12 = dataframes[13 - 2]
df_13 = dataframes[14 - 2]
df_14 = dataframes[15 - 2]
df_15 = dataframes[16 - 2]
df_16 = dataframes[17 - 2]
df_17 = dataframes[18 - 2]
df_18 = dataframes[19 - 2]
df_19 = dataframes[20 - 2]
df_20 = dataframes[21 - 2]

df_lst = []
for i in range(1, 21):
    df_i = globals().get(f'df_{i}')
    if df_i is not None:
        df_lst.append(df_i)

# Concatenate all DataFrames into one big DataFrame
big_df = pd.concat(dataframes, axis=1)


# Define a list to store the results for all "Tropfen"
fill_value = 0
lst = []
results = []

for i in range(1, 21):  # Loop from 1 to 20
    T_i = globals().get(f'T{i}')
    if T_i is not None:
        # Calculate v and dv for each T column
        T_u = np.array((T_i)[:, 0])
        T_d = np.array((T_i)[:, 1])
        T_o = np.array((T_i)[:, 2])
        
        # Create masked arrays
        T_u_masked = ma.masked_equal(T_u, 0)
        T_d_masked = ma.masked_equal(T_d, 0)
        T_o_masked = ma.masked_equal(T_o, 0)

        # Calculate v_T_u, v_T_d, v_T_o
        v_T_u = s / T_u_masked
        v_T_d = s / T_d_masked
        v_T_o = s / T_o_masked

        # Calculate dv_T_u, dv_T_d, dv_T_o
        dv_T_u = np.sqrt((ds / T_u_masked) ** 2 + ((s * dt) / (T_u_masked ** 2)) ** 2)
        dv_T_d = np.sqrt((ds / T_d_masked) ** 2 + ((s * dt) / (T_d_masked ** 2)) ** 2)
        dv_T_o = np.sqrt((ds / T_o_masked) ** 2 + ((s * dt) / (T_o_masked ** 2)) ** 2)
        
        # calculate the means
        v_T_u_mean = np.mean(v_T_u)
        v_T_d_mean = np.mean(v_T_d)
        v_T_o_mean = np.mean(v_T_o)
        dv_T_u_mean = np.mean(dv_T_u)
        dv_T_d_mean = np.mean(dv_T_d)
        dv_T_o_mean = np.mean(dv_T_o)
                       
        #add all data to a list               
        lst.append(np.array([T_u, T_d, T_o, 
                             v_T_u.filled(fill_value), v_T_d.filled(fill_value), v_T_o.filled(fill_value), 
                             dv_T_u.filled(fill_value), dv_T_d.filled(fill_value), dv_T_o.filled(fill_value), 
                             v_T_u_mean, v_T_d_mean, v_T_o_mean, 
                             dv_T_u_mean, dv_T_d_mean, dv_T_o_mean], dtype=object))
        
droplets_array = np.array(lst)

#calculate r and Ne and their errors 
for i in range(20):
    #print(f'droplets_array[{i}][10] = {droplets_array[i][10]}, droplets_array[{i+1}][9] = {droplets_array[i+1][9]}, droplets_array[{i}][13] = {droplets_array[i][13]}, droplets_array[{i}][12] = {droplets_array[i][12]}')   
    if droplets_array[i][10] != 0 and droplets_array[i][9] != 0 and droplets_array[i][13] != 0 and droplets_array[i][12] !=0:
        r = np.sqrt((9 * eta * (droplets_array[i][10] - droplets_array[i][9])) / (4 * g * (oel - luft)))
        #print(r)
        dr = np.sqrt(((3 * (droplets_array[i][10] - droplets_array[i][9])) * deta)/(4 * g * (oel - luft) * np.sqrt(((droplets_array[i][10] - droplets_array[i][9])) * eta) / (g * (oel - luft)))) ** 2 + ((3 * eta * droplets_array[i][13]) / (4 * g * (oel - luft) * np.sqrt(((droplets_array[i][10] - droplets_array[i][9]) * eta) / (g * (oel - luft))))) ** 2 + ((3 * eta * droplets_array[i][12]) / (4 * g * (oel - luft) * np.sqrt(((droplets_array[i][10] - droplets_array[i][9]) * eta) / (g * (oel - luft))))) ** 2
        #print(dr)
        Ne = 3 * np.pi * eta * r * ((droplets_array[i][10] + droplets_array[i][9]) / E)
        #print(Ne)
        dNe = np.sqrt(((3 * np.pi * eta * (droplets_array[i][10] + droplets_array[i][9]) * dr) / E) ** 2 + ((3 * np.pi * eta * r * droplets_array[i][13]) / E) ** 2 + ((3 * np.pi * eta * r * droplets_array[i][12]) / E) ** 2 + ((3 * np.pi * eta * r * (droplets_array[i][10] + droplets_array[i][9]) * dE) / E ** 2) ** 2 + ((3 * np.pi * r * (droplets_array[i][10] + droplets_array[i][9]) * deta) / E) ** 2)
        #print(dNe)
        
    else:
        r = 0
        dr = 0
        Ne = 0
        dNe = 0
        
    results.append(np.array([r, dr, Ne, dNe]))
    
calc_val_array = np.array(results)

# Now, you have a single DataFrame 'big_df' containing all the individual DataFrames
#print(big_df)


# In[48]:


Beschriftung = ['r', 'dr', 'Ne', 'dNe']

Werte = [calc_val_array[i].tolist() for i in range(20)]
gcd = [calc_val_array[i][2] / (1.6 * 10**(-19)) for i in range(20)]
e = [calc_val_array[i][2] / np.round(gcd[i], 0) for i in range(20)]
de = [calc_val_array[i][3] / np.round(gcd[i], 0) for i in range(20)]
N = [np.round(gcd[i], 0) for i in range(20)]

g_names = [f'Tropfen {i}' for i in range(1, 21)]
e_names = [f'Tropfen {i}' for i in range(1, 21)]
de_names = [f'Tropfen {i}' for i in range(1, 21)]
N_names = [f'Tropfen {i}' for i in range(1, 21)]

g_dict = {g_names[i]: Werte[i] for i in range(20)}
e_dict = {e_names[i]: [e[i]] for i in range(20)}
de_dict = {de_names[i]: [de[i]] for i in range(20)}
N_dict = {N_names[i]: [N[i]] for i in range(20)}

g_Werte = pd.DataFrame(g_dict)
g_Werte.insert(0, '###', Beschriftung)

e_Werte = pd.DataFrame(e_dict)
e_Werte.insert(0, '###', ['e'])

de_Werte = pd.DataFrame(de_dict)
de_Werte.insert(0, '###', ['de'])

N_Werte = pd.DataFrame(N_dict)
N_Werte.insert(0, '###', ['N'])


werte_app_1 = g_Werte.append(e_Werte, ignore_index=True)
werte_app_2 = werte_app_1.append(de_Werte, ignore_index=True)
werte_app_3 = werte_app_2.append(N_Werte, ignore_index=True)


new_e = [e[i] ** (2/3) for i in range(20)]
new_de = [(2/3) * e[i] ** (-1/3) * de[i] for i in range(20)]

new_eWerte = np.array(new_e)
new_deWerte = np.array(new_de)

new_r = [1 / calc_val_array[i][0] for i in range(20)]
new_dr = [calc_val_array[i][1] / calc_val_array[i][0] ** 2 for i in range(20)]

new_rWerte = np.array(new_r)
new_drWerte = np.array(new_dr)

Cunningham_Korrektur = []

def x_ges(x, n):
    return (1 / n) * np.sum(x)

def sigma(val, n):
    return n / (np.sum(1 / val ** 2))

def polynom(m, b, x):
    return m * x + b

def geradenfit(x, y, x_err, y_err, n):
    x_strich = x_ges(x, n)
    x2_strich = x_ges(x ** 2, n)
    y_strich = x_ges(y, n)
    xy_strich = x_ges(x * y, n)
    print(f'{x_strich = }')
    print(f'{y_strich = }')
    print(f'{x2_strich =}')
    print(f'{xy_strich = }')

    m = (xy_strich - (x_strich * y_strich)) / (x2_strich - x_strich ** 2)
    b = (x2_strich * y_strich - x_strich * xy_strich) / (x2_strich - x_strich ** 2)
    print(f'{m = }')
    print(f'{b = }')

    sigmax = sigma(x_err, n)
    sigmay = sigma(y_err, n)

    dm = np.sqrt(sigmay / (n * (x2_strich - x_strich ** 2)))
    db = np.sqrt(sigmay * x2_strich / (n * (x2_strich - (x_strich ** 2))))
    print(f'{dm = }')
    print(f'{db = }')

    fig, ax = plt.subplots()
    ax.set_title('Cunningham-Korrektur')
    ax.set_ylabel(r'$e^{\frac{2}{3}}$[C]')
    ax.set_xlabel(r'$\frac{1}{r_i}$[1/m]')
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, capsize=2, fmt='.', markersize=5, color='black')
    ax.plot(x, polynom(m, b, x), label=f'$y = ({m:0.3e})x+({b:+0.3e})$')
    plt.legend()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
    plt.savefig('Cunningham-Korrektur.png', dpi=300)
    plt.show()

    e0 = b ** (3 / 2)
    de0 = 3 / 2 * db * e0 ** (1 / 2)

    print(f'{e0 = }')
    print(f'{de0 = }')

    em = 163844258410.99548
    dem = 514943367.8875455
    me = e0 / em
    dme = np.sqrt((de0 / em) ** 2 + ((e0 * dem) / em ** 2) ** 2)

    print(f'{me = }')
    print(f'{dme = }')
    
    items_to_add = [x_strich, y_strich, x2_strich, xy_strich, 
                    m, b, dm, db, e0, de0, me, dme]
    Cunningham_Korrektur[:] = [*Cunningham_Korrektur, *items_to_add]

geradenfit(new_rWerte, new_eWerte, new_drWerte, new_deWerte, 20)

Cunningham_df = pd.DataFrame(np.array(Cunningham_Korrektur))
Cunningham_df.insert(0, '###', ['x_strich', 'y_strich', 'x2_strich', 'xy_strich', 
                    'm', 'b', 'dm', 'db', 'e0', 'de0', 'me', 'dme'])


# In[50]:


#print and save dataframes and plots

#g_Werte.to_excel('g_Werte.xlsx', sheet_name='sheet1!', index=False)
#e_Werte.to_excel('e_Werte.xlsx', sheet_name='sheet1!', index=False)
#de_Werte.to_excel('de_Werte.xlsx', sheet_name='sheet1!', index=False)
#N_Werte.to_excel('N_Werte.xlsx', sheet_name='sheet1!', index=False)
#werte_app_3.to_excel('alle_Werte.xlsx', sheet_name='sheet1!', index=False)
#big_df.to_excel('Millikan.xlsx', sheet_name='sheet1!', index = False)
#Angaben_df.to_excel('Angaben-Messinstrument.xlsx', sheet_name='sheet1!', index=False) 
#Cunningham_df.to_excel('Cunningham.xlsx', sheet_name='sheet1!', index=False)

#werte_app_3

