#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from kafe2 import XYContainer, Fit, Plot


# In[2]:


#Hier weren die Daten geladen

input_file_path_1 = "./Versuch240b.txt"
output_file_path_1 = "240b.txt"

# Open the input file for reading
with open(input_file_path_1, "r") as input_file:
    # Read all lines from the input file
    lines_1 = input_file.readlines()

# Remove the first 5 rows by slicing the list
lines_1 = lines_1[16:]

# Replace commas with periods in each line
lines_1 = [line.replace(",", ".") for line in lines_1]

# Open the output file for writing
with open(output_file_path_1, "w") as output_file:
    # Write the modified lines to the output file
    output_file.writelines(lines_1)

Messdaten1 = "240b.txt" #Dateiname

#Daten Laden und ausgeben
data1 = np.loadtxt(Messdaten1)


zero_count = 0  # Initialize the count of zeros
split_index = None  # Initialize the split index to None

data_1 = data1[:549]
    
#Spalten der Datei auslesen
column1 = data_1[:,0] #1.Spalte
column2 = data_1[:,1] #2.Spalte
column3 = data_1[:,2] #3.Spalte


# Werte für Neukurve splitten
max_current = float('-inf')  # Initialize max_current to negative infinity
split_index1 = None  # Initialize the split index to None

# Iterate through the data to find the index where the maximum current is reached
for index, sub_array in enumerate(data1):
    if sub_array[1] > max_current:
        max_current = sub_array[1]
        split_index1 = index

# Check if a split point was found
if split_index1 is not None:
    # Split the data into two arrays
    nc_1 = data1[:split_index1 + 1]
    nc_2 = data1[split_index1 + 1:]
else:
    # If no maximum current was found, the data remains in a single array
    nc_1 = data1
    nc_2 = []

#print("Array split 1:", nc_1)
#print("Array split 2:", nc_2)




# Werte für mhy_A splitten
ma_1 = data1[2:9]
ma_2 = data1[9:]
#print(ma_1)
#print(ma_2)

#print(ma_1)
ma_1


# In[3]:


df_1 = pd.DataFrame(data1)

df_data1 = pd.DataFrame(data_1)

df_nc_1 = pd.DataFrame(nc_1)

df_ma_1 = pd.DataFrame(ma_1)


# In[4]:


#Daten für Neukurve

#Spalten der Datei auslesen
new_curve_column1 = nc_1[:,0] #1.Spalte
new_curve_column2 = nc_1[:,1] #2.Spalte
new_curve_column3 = nc_1[:,2] #3.Spalte


# In[5]:


#My_A_Fit

#Spalten der Datei auslesen (funktioniert nur, wenn da genau richtig viele column stehen)
my_a_column1 = ma_1[:,0] #1.Spalte
my_a_column2 = ma_1[:,1] #2.Spalte
my_a_column3 = ma_1[:,2] #3.Spalte

#definiere N als Anzahl der Messwertpaare
my_a_N=len(my_a_column1)
print("length comumb1:", len(my_a_column1))
print("length comumb2:", len(my_a_column2))
print("length comumb3:", len(my_a_column3))
print("N:", my_a_N)

ma_x_data = my_a_column2
ma_y_data = my_a_column3

print("x_data:" , ma_x_data)
print("y_data:" , ma_y_data)


# In[6]:


#240_b

#gemessene Werte
t = column1
#print("Zeit t:", t)
#print("Lenght t:", len(t))

I = column2
#print("Stromstaerke I:", I)
#print("Length I:", len(I))

B = (-column3)/1000 #umrechnung in Tesla (SI-Einweit, damit die Formeln stimmen) und Graph an x-Achse spiegeln
#print("Magnetische Flussdichte B:", B)
#print("Lenght B:", len(B))

#gegebene Werte für die Formlen:
N = 2*500 #Gesammtwindungszahl aus 2 Spule zusammengesetzt
l_Fe = 0.477 #Länge eines Weges durch die Mitte des Eisenkerns in Meter (Laut Anleitung: 477 +- 4mm)
My_0 = 1.25663706212e-6 #in N/A^-2 , Literaturwert https://physics.nist.gov/cgi-bin/cuu/Value?mu0 (Stand 02.11.2022)
d = 0.002 #Dicke des Spaltes in Meter (Laut Anleitung: 2 +- 0,05 mm)

#Berechnung der magnetischen Feldstärke
H = ((N*I)/l_Fe)-(d/(My_0*l_Fe))*B #Formel: 240.2 S.50
#print("Magnetische Feldstärke H:", H)

#Fehlerrechnung
N_err = 0 #0, da Natürliche Zahl
#print("Fehler auf N:", N_err)
l_Fe_err=4/1000 # in Meter
#print("Fehler auf l_Fe:", l_Fe_err)
My_0_err= 1.9*10**(-10)*10**(-6) #in N/A^-2 , Literaturwert https://physics.nist.gov/cgi-bin/cuu/Value?mu0 (Stand 02.11.2022)
#print("Fehler auf My_0:", My_0_err)
d_err=0.05/1000 # in Meter
#print("Fehler auf d:", d_err)
I_err=I*0.01 #1% des Messwertes
#print("Fehler auf I:", I_err)
B_err=B*0.03 #3% des Messwertes bei 20°C
#print("Fehler auf B:", B_err)

#Fehlerquadrate der einzelnen Ableitungen(Teilergebnisse für H_err)
s1 = (N/l_Fe)*I_err #H nach I ableiten
s2 = (-B/(My_0*l_Fe))*d_err #H nach d ableiten
s3 = (-d/(My_0*l_Fe))*B_err #H nach B ableiten
s4 = ((B*d-I*My_0*N)/(My_0*l_Fe**2))*l_Fe_err #H nach l_Fe ableiten
s5 = ((d)/(My_0**2*l_Fe))*My_0_err #H nach My0 ableiten


#Fehler auf die magnetische Fledstärke
H_err = (s1**2+s2**2+s3**2+s4**2+s5**2)**(1/2)
#print("Fehler auf H:", H_err)


# In[7]:


def merge_measurement_data(t, I, I_err, B, B_err, H, H_err, N, N_err, l_Fe, l_Fe_err, My_0, My_0_err, d, d_err):
    df_t = pd.DataFrame(t)
    df_I = pd.DataFrame(I)
    df_I_err = pd.DataFrame(I_err)
    df_B = pd.DataFrame(B)
    df_B_err = pd.DataFrame(B_err)
    df_H = pd.DataFrame(H)
    df_H_err = pd.DataFrame(H_err)
    df_N = pd.DataFrame(np.repeat(N, len(I)))
    df_N_err = pd.DataFrame(np.repeat(N_err, len(I)))
    df_l_Fe = pd.DataFrame(np.repeat(l_Fe, len(I)))
    df_l_Fe_err = pd.DataFrame(np.repeat(l_Fe_err, len(I)))
    df_My_0 = pd.DataFrame(np.repeat(My_0, len(I)))
    df_My_0_err = pd.DataFrame(np.repeat(My_0_err, len(I)))
    df_d = pd.DataFrame(np.repeat(d, len(I)))
    df_d_err = pd.DataFrame(np.repeat(d_err, len(I)))

    merged_df = df_t.reset_index().merge(df_I.reset_index(), on='index')
    merged_df = merged_df.merge(df_I_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_B.reset_index(), on='index')
    merged_df = merged_df.merge(df_B_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_H.reset_index(), on='index')
    merged_df = merged_df.merge(df_H_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_N.reset_index(), on='index')
    merged_df = merged_df.merge(df_N_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_l_Fe.reset_index(), on='index')
    merged_df = merged_df.merge(df_l_Fe_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_My_0.reset_index(), on='index')
    merged_df = merged_df.merge(df_My_0_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_d.reset_index(), on='index')
    merged_df = merged_df.merge(df_d_err.reset_index(), on='index')

    return merged_df


# In[8]:


#Neukurve

#gemessene Werte
nc_t = new_curve_column1
#print("Zeit t:", nc_t)
#print("Lenght t:", len(t))

nc_I = new_curve_column2
#print("Stromstaerke I:", I)
#print("Length I:", len(I))

nc_B = (-new_curve_column3)/1000 #umrechnung in Tesla (SI-Einweit, damit die Formeln stimmen) und Graph an x-Achse
#print("Magnetische Flussdichte B:", B)
#print("Lenght B:", len(B))

#Berechnung der magnetischen Feldstärke
nc_H = ((N*nc_I)/l_Fe)-(d/(My_0*l_Fe))*nc_B #Formel: 240.2 S.50
#print("Magnetische Feldstärke H:", nc_H)

#Fehlerrechnung

nc_I_err=nc_I*0.01 #1% des Messwertes
#print("Fehler auf I:", nc_I_err)
nc_B_err=nc_B*0.03 #3% des Messwertes bei 20°C
#print("Fehler auf B:", nc_B_err)

#Fehlerquadrate der einzelnen Ableitungen(Teilergebnisse für H_err)
nc_s1 = (N/l_Fe)*nc_I_err #H nach I ableiten
nc_s2 = (-nc_B/(My_0*l_Fe))*d_err #H nach d ableiten
nc_s3 = (-d/(My_0*l_Fe))*nc_B_err #H nach B ableiten
nc_s4 = ((nc_B*d-nc_I*My_0*N)/(My_0*l_Fe**2))*l_Fe_err #H nach l_Fe ableiten
nc_s5 = ((d)/(My_0**2*l_Fe))*My_0_err #H nach My0 ableiten


#Fehler auf die magnetische Fledstärke
nc_H_err = (nc_s1**2+nc_s2**2+nc_s3**2+nc_s4**2+nc_s5**2)**(1/2)
#print("Fehler auf H:", nc_H_err)


# In[9]:


def merge_measurement_nc_data(nc_t, nc_I, nc_I_err, nc_B, nc_B_err, nc_H, nc_H_err):
    df_nc_t = pd.DataFrame(nc_t)
    df_nc_I = pd.DataFrame(nc_I)
    df_nc_I_err = pd.DataFrame(nc_I_err)
    df_nc_B = pd.DataFrame(nc_B)
    df_nc_B_err = pd.DataFrame(nc_B_err)
    df_nc_H = pd.DataFrame(nc_H)
    df_nc_H_err = pd.DataFrame(nc_H_err)

    merged_df = df_nc_t.reset_index().merge(df_nc_I.reset_index(), on='index')
    merged_df = merged_df.merge(df_nc_I_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_nc_B.reset_index(), on='index')
    merged_df = merged_df.merge(df_nc_B_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_nc_H.reset_index(), on='index')
    merged_df = merged_df.merge(df_nc_H_err.reset_index(), on='index')

    return merged_df


# In[10]:


#240_c

#gemessene Werte
ma_t = my_a_column1
#print("Zeit t:", ma_t)
#print("Lenght t:", len(t))

ma_I = my_a_column2
#print("Stromstaerke I:", ma_I)
#print("Length I:", len(I))

ma_B = (-my_a_column3)/1000 #umrechnung in Tesla (SI-Einweit, damit die Formeln stimmen) und Graph an x-Achse
#print("Magnetische Flussdichte B:", ma_B)
#print("Lenght B:", len(B))

#Berechnung der magnetischen Feldstärke
ma_H = ((N*ma_I)/l_Fe)-(d/(My_0*l_Fe))*ma_B #Formel: 240.2 S.50
#print("Magnetische Feldstärke H:", ma_H)

ma_I_err=ma_I*0.01 #1% des Messwertes
#print("Fehler auf I:", ma_I_err)
ma_B_err=ma_B*0.03 #3% des Messwertes bei 20°C
#print("Fehler auf B:", ma_B_err)

#Fehlerquadrate der einzelnen Ableitungen(Teilergebnisse für H_err)
ma_s1 = (N/l_Fe)*ma_I_err #H nach I ableiten
ma_s2 = (-ma_B/(My_0*l_Fe))*d_err #H nach d ableiten
ma_s3 = (-d/(My_0*l_Fe))*ma_B_err #H nach B ableiten
ma_s4 = ((ma_B*d-ma_I*My_0*N)/(My_0*l_Fe**2))*l_Fe_err #H nach l_Fe ableiten
ma_s5 = ((d)/(My_0**2*l_Fe))*My_0_err #H nach My0 ableiten


#Fehler auf die magnetische Fledstärke
ma_H_err = (ma_s1**2+ma_s2**2+ma_s3**2+ma_s4**2+ma_s5**2)**(1/2)
#print("Fehler auf H:", ma_H_err)

x_data = ma_H
y_data = ma_B
x_err = ma_H_err
y_err = ma_B_err


# In[11]:


def merge_measurement_ma_data(ma_t, ma_I, ma_I_err, ma_B, ma_B_err, ma_H, ma_H_err):
    df_ma_t = pd.DataFrame(ma_t)
    df_ma_I = pd.DataFrame(ma_I)
    df_ma_I_err = pd.DataFrame(ma_I_err)
    df_ma_B = pd.DataFrame(ma_B)
    df_ma_B_err = pd.DataFrame(ma_B_err)
    df_ma_H = pd.DataFrame(ma_H)
    df_ma_H_err = pd.DataFrame(ma_H_err)

    merged_df = df_ma_t.reset_index().merge(df_ma_I.reset_index(), on='index')
    merged_df = merged_df.merge(df_ma_I_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_ma_B.reset_index(), on='index')
    merged_df = merged_df.merge(df_ma_B_err.reset_index(), on='index')
    merged_df = merged_df.merge(df_ma_H.reset_index(), on='index')
    merged_df = merged_df.merge(df_ma_H_err.reset_index(), on='index')

    return merged_df


# In[12]:


#Hier wird der Plot erstellt

plt.rc ('font', size = 20) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = 20) # Schriftgröße des Titels
plt.rc ('axes', labelsize = 20) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = 20) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = 20) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = 20) #Schriftgröße der Legende

#Plot ohne Fehlerbalken
plt.figure(figsize=(20,20)) #Größe des Bildes
plt.scatter(H,B) #eigentlicher Plot
plt.xlabel("magnetische Fledstärke H in A/m") #Achsenbeschriftung x-Achse
plt.ylabel("magnetische Flussdichte B in T") #Achsenbeschriftung y-Achse
plt.title("Hysteresekurve ohne Fehlerbalken") #Titel des Plots
plt.show #Plot anzeigen
plt.savefig('240c_ohne_Fehlerbalken.jpg', dpi = 300) #Plot als Datei abspeichern


# In[13]:


#Plot mir Fehlerbalken
plt.figure(figsize=(20,20)) #Größe des Bildes
plt.scatter(H,B) #eigentlicher Plot
plt.xlabel("magnetische Fledstärke H in A/m") #Achsenbeschriftung x-Achse
plt.ylabel("magnetische Flussdichte B in T") #Achsenbeschriftung y-Achse
plt.title("Hysteresekurve mit Fehlerbalken") #Titel des Plots
plt.errorbar(H, B, xerr=H_err, yerr=np.absolute(B_err), fmt="o", color="r") #Fehlerbalken, color r = Rot
plt.show #Plot anzeigen
plt.savefig('240c_mit_Fehlerbalken.jpg', dpi = 300) #Plot als jpg-Datei abspeichern


# In[14]:


plt.rc ('font', size = 20) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = 20) # Schriftgröße des Titels
plt.rc ('axes', labelsize = 20) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = 20) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = 20) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = 20) #Schriftgröße der Legende

Steigung = 4.31*10**(-4)
Steigung_max = 4.66*10**(-4)
Steigung_min = 3.96*10**(-4)

#Plot ohne Fehlerbalken
plt.figure(figsize=(20,20)) #Größe des Bildes
plt.scatter(nc_H,nc_B) #eigentlicher Plot
plt.plot(nc_H, Steigung*nc_H, linestyle='solid', color = "r") #Tangente zur Bestimmung von My_max

plt.plot(nc_H, Steigung_max*nc_H, linestyle='solid', color = "g")
plt.plot(nc_H, Steigung_min*nc_H, linestyle='solid', color = "g")

plt.xlabel("magnetische Fledstärke H in A/m") #Achsenbeschriftung x-Achse
plt.ylabel("magnetische Flussdichte B in T") #Achsenbeschriftung y-Achse
plt.title("Neukurve ohne Fehlerbalken") #Titel des Plots
plt.show #Plot anzeigen
plt.savefig('240cNeukurve_ohne_Fehlerbalken.jpg', dpi = 300) #Plot als Datei abspeichern

#Plot mir Fehlerbalken
plt.figure(figsize = (20,20)) #Größe des Bildes
plt.scatter(nc_H,nc_B) #eigentlicher Plot
plt.plot(nc_H, Steigung*nc_H, linestyle='solid', color = "b") #Tangente zur Bestimmung von My_max

plt.plot(nc_H, Steigung_max*nc_H, linestyle='solid', color = "g")
plt.plot(nc_H, Steigung_min*nc_H, linestyle='solid', color = "g")

plt.xlabel("magnetische Fledstärke H in A/m") #Achsenbeschriftung x-Achse
plt.ylabel("magnetische Flussdichte B in T") #Achsenbeschriftung y-Achse
plt.title("Neukurve mit Fehlerbalken") #Titel des Plots
plt.errorbar(nc_H, nc_B, xerr = nc_H_err, yerr = nc_B_err, fmt = "o", color = "r") #Fehlerbalken, color r = Rot
plt.show() #Plot anzeigen
plt.savefig('240cNeukurve_mit_Fehlerbalken.jpg', dpi = 300) #Plot als jpg-Datei abspeichern

#My_max berechnen
My_max = Steigung/My_0
print("My_max:", My_max)

#Fehlerrechnung
Steigung_err =  0.4*10**(-4) #geschätzt
s1 = (1/My_0)*Steigung_err
s2 = (-Steigung/My_0**2)*My_0_err
My_max_err = (s1**2+s2**2)**(1/2)
print("Fehler auf My_max:", My_max_err)


# In[15]:


#N = Anzahl der Wertepaare
print("Anzahl der Wertepaare N:" , my_a_N)

#Umbenennen der Variablen
x=x_data
y=y_data
# Fehler x_err und y_err werden aus der vorherigen zelle übernommen

#Bildung der Varianzgewichteten Mittelwerte z ∈ {x, y, xy, x^2}

#Bildung von Zähler und Nenner der Rechnungen z/y_err**2
sumyerr  = sum(1/(y_err)**2)
#print("Dies ist der Nenner der Mittelwerte:", sumyerr)
xdivyerr = sum(x/(y_err)**2)
#print("Dies ist der Zähler des x-Mittelwert:", xdivyerr)
ydivyerr = sum(y/(y_err)**2)
#print("Dies ist der Zähler des y-Mittelwert:", ydivyerr)
x2divyerr = sum((x**2)/(y_err)**2)
#print("Dies ist der Zähler des x^2-Mittelwert:", x2divyerr)
xydivyerr = sum((x*y)/(y_err)**2)
#print("Dies ist der Zähler des xy-Mittelwert:", xydivyerr)

#Bildung der Varianzgewichteten Mittelwerte
xMittel = xdivyerr/sumyerr
#print("Dies ist der varianzgewichteter Mittelwert von X: ",xMittel)
yMittel = ydivyerr/sumyerr
#print("Dies ist der varianzgewichteter Mittelwert von Y: ",yMittel)
x2Mittel = x2divyerr/sumyerr
#print("Dies ist der Mittelwert von X^2: ",x2Mittel)
xyMittel = xydivyerr/sumyerr
#print("Dies ist der Mittelwert von XY: ",xyMittel)


#Berechnung von m und n:
m = (xyMittel-(xMittel*yMittel))/(x2Mittel-(xMittel)**2) #Steigung m
print("Dies ist die Steigung m: ",m)
n = yMittel - m*xMittel #Achsenabschnitt n
print("Dies ist der y-Achsenabschnitt n: ",n)

#Berechnung von den Varianzen:
sigmayMittel = N/sumyerr #Varianzgewichtete Standardabweichung Sigmay
#print("Dies ist die Varianzgemittelte Standardabweichung Sigmay: ", sigmayMittel)
Vm = sigmayMittel/(16*(x2Mittel-(xMittel)**2)) #Varianz der Steigung
print("Dies ist die Varianz auf m (V[m]): ",Vm)
Vn = x2Mittel*Vm #Varianz des Achsenabschnitt
print("Dies ist die Varianz auf n (V[n]): ",Vn)
Vmn = -Vm * xMittel #Kovarianz von m und n
print("Dies ist die Kovarianz von m und n (Vmn): ",Vmn)

#Berechnung der Güte:
xi2array = (y-m*x-n)**2/y_err #
xi2 = sum(xi2array)
#print("Dies ist der xi^2-Wert des Fits: ",xi2)
güte = xi2/(14-2) #Güte
print("Wir erhalten die Güte als: ", güte)


def linearf(x, m, b):# lineare Funktion aufstellen
    return m*x+b
def linfit(x_data, y_data, x_err, y_err): # funktion mit den daten als argumenten erstellen
    xy_data = XYContainer(x_data, y_data) # behälter für x und y erstellen
    xy_data.add_error(axis='x', err_val=x_err) # x-Fehlerbalken
    xy_data.add_error(axis='y', err_val=y_err) # y-Fehlerbalken
    linear_fit = Fit(data=xy_data, model_function=linearf) # Objekt mit dem xy- Behälter und der linearen Funkrion erzeugen
    fit_results = linear_fit.do_fit() # ergebnisse bestimmen und später ausgeben lassen
    #linear_fit.report()
    plot = Plot(fit_objects=linear_fit) 
    plot.plot()
    
    #plot als jpg abspeichern und anzeigen
    plt.savefig('240c_My_A_Fit.jpg', dpi = 300)
    plt.show() #Plot anzeigen
    return fit_results

# Ergebnisse in der Kovarianzmatrix ausgeben lassen
print(linfit(x_data, y_data, x_err, y_err)['parameter_cov_mat'])


#My_A berechnen
My_A = m/My_0
print("My_A:", My_A)

#Fehlerrechnung
s1 = (1/My_0)*Vm
s2 = (-m/My_0**2)*My_0_err
My_A_err = (s1**2+s2**2)**(1/2)
print("Fehler auf My_A:", My_A_err)


# In[16]:


def save_dataframe_to_excel(dataframe, file_path):
    """
    Save a DataFrame to an Excel file.

    Parameters:
    - dataframe: The DataFrame to be saved.
    - file_path: The path to the Excel file (e.g., 'output.xlsx').
    """
    try:
        dataframe.to_excel(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error: {e}")


# In[30]:


# Measurement 1
merged_b1 = merge_measurement_data(t, I, I_err, B, B_err, H, H_err, N, N_err, l_Fe, l_Fe_err, My_0, My_0_err, d, d_err).set_index('index')
merged_nc1 = merge_measurement_nc_data(nc_t, nc_I, nc_I_err, nc_B, nc_B_err, nc_H, nc_H_err).set_index('index')
merged_ma1 = merge_measurement_ma_data(ma_t, ma_I, ma_I_err, ma_B, ma_B_err, ma_H, ma_H_err).set_index('index')

merged_b1.to_excel('data.xlsx', sheet_name='sheet1!', index=False)
merged_nc1.to_excel('newcurve.xlsx', sheet_name='sheet1!', index=False)
merged_ma1.to_excel('myA.xlsx', sheet_name='sheet1!', index=False)

