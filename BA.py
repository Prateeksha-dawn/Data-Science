

#from sklearn.linear_model import LinearRegression

class BA():
    def __init__(self,dep,indep):
        import numpy as np
        self.dep = np.asarray(dep)
        self.indep =np.asarray(indep)
        self.mean      = np.mean([self.dep, self.indep], axis=0)
        self.diff      = self.dep - self.indep                   # Difference between data1 and data2
        self.md        = np.mean(self.diff)            # Mean of the difference
        self.sd        = np.std(self.diff, axis=0) # Standard deviation of the difference
        self.UL        = self.md + 1.96*self.sd
        self.LL        = self.md - 1.96*self.sd
        self.Normality()
        
    def Normality(self):
        import scipy.stats
        from scipy.stats import shapiro
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        import numpy as np
        
        ##### NORMAILY CHECK :
        stat, p = shapiro(self.diff)
        print('Statistics=%.3f, p=%.3f' % (stat, p))     # interpret
        alpha = 0.05

        if p > alpha:
            print('The difference Between two methods is normaly distributed')  

        else:
            print('The difference between two methods is not normally distributed')  


        ######## HISTOGRAM:
        self.mu, self.std = norm.fit(self.diff)
        fig=plt.hist(self.diff, bins=25, density=True, alpha=0.6, color='grey')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, self.mu, self.std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (self.mu, self.std)
        plt.title(title)
        plt.xlabel("Difference")
        plt.show()
        self.LinearRegression()
        
    def LinearRegression(self):
        import matplotlib.pyplot as plt
        import numpy as np
        ####### LINEAR REGRESSION:
        n = np.size(self.mean)     # number of observations/points 
        m_x, m_y = np.mean(self.mean), np.mean(self.diff)      # mean of x and y vector 

        SS_xy = np.sum(self.diff*self.mean) - n*m_y*m_x  # calculating cross-deviation and deviation about x 
        SS_xx = np.sum(self.mean*self.mean) - n*m_x*m_x 

        b_1 = SS_xy / SS_xx      # calculating regression coefficients 
        b_0 = m_y - b_1*m_x 

        plt.scatter(self.mean, self.diff, color = "black", 
                   marker = "o", s = 12) 

        self.y_pred = [b_0] + [b_1]*self.mean   # predicted response vector 

        plt.plot(self.mean, self.y_pred, color = "grey")      # plotting the regression line
        plt.title("Linear Regression")

        plt.xlabel('mean')   # putting labels 
        plt.ylabel('difference') 
        plt.show()  # function to show plot 
        self.Bland_Altman_plot()
        
    def Bland_Altman_plot(self,*args, **kwargs):
        import numpy as np
        import matplotlib.pyplot as plt
         ####### BLAND ALTMAN SCATTER PLOT             

        plt.figure(figsize=(8, 5), dpi=80)
        plt.scatter(self.mean, self.diff, *args, **kwargs,s=3,facecolors='black')
        plt.axhline(self.md, color='black', linestyle='--')
        plt.axhline(self.UL, color='black', linestyle='-')
        plt.axhline(self.LL, color='black', linestyle='-')

        max_y=(np.max(self.diff)+10)
        plt.ylim(-(max_y),+(max_y))


        min_x=(min(self.mean))

        plt.text(min_x,(self.md+0.5),("BIAS=%.3f"%(self.md)),fontweight='bold', fontsize= 10,ha='left', va='center')
        plt.text(min_x,(self.UL+0.5),("UPPER LOA=%.3f"%(self.UL)),fontweight='bold', fontsize=10,ha='left', va='center')
        plt.text(min_x,(self.LL+0.5),("LOWER LOA=%.3f"%(self.LL)),fontweight='bold', fontsize=10,ha='left', va='center')

        print ("whats the title?")
        subtitle= input()
        print ("Total sample size")
        title = "Total="+input()
        plt.suptitle(subtitle, y=1, fontsize=18)
        plt.title(title, y=1.02, fontsize=10,loc='right',ha='center', va='center')


        print("X-AXIS ?")#mean
        plt.xlabel(input())
        print("Y-AXIS ?")#difference
        plt.ylabel(input())
        plt.show()
        self.Agreement_percentage()
        
    def Agreement_percentage(self):
        import scipy.stats
        import numpy as np
            ####### AGREEMENT PERCENTAGE:
        count= np.logical_and(self.diff<=self.UL,self.LL)
        a1=sum(count)
        a2=len(self.diff)
        z=round(((a1)/(a2))*100,2)
        print("Percentage of values that lie between the Upper and Lower Limit "+str(z)+"%")

        REG=scipy.stats.linregress(self.mean,self.diff)
        print(REG)

        




        


        
        
