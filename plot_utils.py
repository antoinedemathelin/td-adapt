# -*- coding: utf-8 -*-
"""
@author: Mounir
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def label_func(x,beta=2,noise=0.1):
    return beta*x + noise * np.random.randn(len(x))

def gaussian(x, mu=0., s=1.):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -(x-mu)**2 / ( 2. * s**2 ) )

def plot_dots(ax,X,Y, title='Source data'):

    inds_0 = np.where(Y==0)[0]
    inds_1 = np.where(Y==1)[0]
    ax.scatter(X[inds_0, 0], X[inds_0, 1],marker='o',edgecolor='black',color='blue',label='class 0')
    ax.scatter(X[inds_1, 0], X[inds_1, 1],marker='o',edgecolor='black',color='red',label='class 1')
    
    ax.set_title(title)
    ax.legend()

    #plt.show()

def plot_dec_func(ax, clf, x_min, x_max, y_min, y_max, plot_step = 0.01, title='Source data'):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
    
    #Source decision function:
    ypred_src = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ypred_src = ypred_src.reshape(xx.shape)

    
    ax.contourf(xx, yy, ypred_src, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.set_title(title)
    #ax.legend()
    #plt.show()

def plot_1d_func(ax, fx, x_min, x_max, y_min, y_max, plot_step = 0.01, title='Source data'):
    xx = np.linspace(x_min,x_max,int((x_max-x_min)/plot_step))
    ypred_src = fx(xx)
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.plot(xx,ypred_src,color='black')
    
def plot_dec_func_explicite(ax, fx, x_min, x_max, y_min, y_max, plot_step = 0.01, title='Source data'):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
    
    #Source decision function:
    ypred_src = fx(xx)
    ypred_src = ypred_src.reshape(xx.shape)
    ypred_src = yy > ypred_src
    
    ax.plot(xx,fx(xx),color='black' )
    ax.contourf(xx, yy, ypred_src, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.set_title(title)
    #ax.legend()
    

def show_reg1d(Xs=None, Xt=None, ys=None, yt=None, model=None, weights=None,mu_s=1,mu_t=-1,sig_s=1,sig_t=1):
    """
    This is the plotting function 
    """

    if ys is None and yt is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))

        alpha = 0.1
        for x in Xs:
            ax.plot([x, x], [0, alpha], c="C0")
        ax.plot([x, x], [0, alpha], c="C0", label=r"$x_i$ observations")
        
        alpha = 0.1
        for x in Xt:
            ax.plot([x, x], [0, alpha], c="C1")
        ax.plot([x, x], [0, alpha], c="C1", label=r"$x'_i$ observations")

        ax.tick_params(left = False, right = False , labelleft = True ,
                        labelbottom = True, bottom = False, top = False)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

        ylim_max = 0.45

        ax.plot(1, 0.0, ls="", marker=">", ms=10, color="k", clip_on=False,
                transform=ax.get_yaxis_transform())
        ax.plot(-4.3, ylim_max, ls="", marker="^", ms=10, color="k", clip_on=False)

        # ax.set_ylim(0., ylim_max)
        ax.set_xlim(-4.3, 4.3)

        ax.set_xlabel("X ~ P(X)", fontsize=16)
        ax.set_ylabel(r"$\widehat{p_s}(x)$", fontsize=16)
        
        if weights is None:
            sns.kdeplot(Xs, label=r"$\widehat{p_s}(x)$", ax=ax, shade=True)
        else:
            np.random.seed(123)
            bs_index = np.random.choice(len(Xs), 3 * len(Xs), p=weights/weights.sum())
            sns.kdeplot(Xs[bs_index], label=r"$w(x) \widehat{p_s}(x)$", ax=ax, shade=True)
        sns.kdeplot(Xt, label=r"$\widehat{p_t}(x)$", ax=ax, shade=True)

        ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1.4, 1.05))
    
    else:
        fig = plt.figure(figsize=(8, 8))

        gs = fig.add_gridspec(2, 1, height_ratios=(2, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.0)
        # Create the Axes.
        ax2 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
        
        if weights is None:
            lns1 = ax1.plot(Xs, ys, '.', ms=15, alpha=0.7, c="C0",
                            markeredgecolor="C0", label=r"$(x_i, y_i)$ observations")
        else:
            lns1 = ax1.scatter(Xs, ys, s=100*weights, alpha=0.7, c="C0",
                            edgecolor="C0", label=r"$(x_i, y_i)$ observations")
            lns1 = [lns1]
        lns8 = ax1.plot(Xt, yt, '.', ms=15, alpha=0.7, c="C1",
                        markeredgecolor="C1", label=r"$(x'_i, y'_i)$ !not available!")

        lin = np.linspace(-4.2, 4.2, 100)

        ax2.plot(lin, gaussian(lin, mu=mu_s, s=sig_s), color="C0")
        lns2 = ax2.fill_between(lin, gaussian(lin, mu=mu_s),
                                alpha=0.2, color="C0", label=r"$p_s(x)$")
        ax2.plot(lin, gaussian(lin, mu=mu_t), color="C1")
        lns6 = ax2.fill_between(lin, gaussian(lin, mu=mu_t, s=sig_t),
                                alpha=0.2, color="C1", label=r"$p_t(x)$")

        alpha = 0.1
        for x in Xs:
            ax2.plot([x, x], [0, alpha], c="C0")
        lns3 = ax2.plot([x, x], [0, alpha], c="C0", label=r"$x_i$ observations")

        alpha = 0.1
        for x in Xt:
            ax2.plot([x, x], [0, alpha], c="C1")
        lns7 = ax2.plot([x, x], [0, alpha], c="C1", label=r"$x'_i$ observations")

        ax2.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = True, bottom = False, top = False)
        ax2.spines.top.set_visible(False)
        ax2.spines.right.set_visible(False)

        ax1.tick_params(left = True, right = False , labelleft = True ,
                        labelbottom = False, bottom = False, top = False)
        ax1.spines.top.set_visible(False)
        ax1.spines.right.set_visible(False)
        ax1.spines.bottom.set_visible(False)

        ylim_max_ax2 = 0.45
        ylim_min_ax2 = 0.
        ylim_max_ax1 = ax1.get_ylim()[1]
        ylim_min_ax1 = ax1.get_ylim()[0]

        if model is not None:   
            if hasattr(model, "predict"):
                yp = model.predict(lin.reshape(-1, 1))
            else:
                yp = model(lin.reshape(-1, 1))
            lns4 = ax1.plot(lin, yp, c="k", label=r"$\widehat{h^*}$  predictive model")

        ax2.plot(1, 0.0, ls="", marker=">", ms=10, color="k", clip_on=False,
                transform=ax2.get_yaxis_transform())
        ax1.plot(-4.3, ylim_max_ax1, ls="", marker="^", ms=10, color="k", clip_on=False)

        ax2.set_ylim(ylim_min_ax2, ylim_max_ax2)
        ax1.set_ylim(ylim_min_ax1, ylim_max_ax1)
        ax2.set_xlim(-4.3, 4.3)
        ax2.set_xlim(-4.3, 4.3)

        ax2.set_xlabel("X ~ P(X)", fontsize=16)
        ax1.set_ylabel("Y ~ P(Y|X)", fontsize=16)

        lns = lns1 + [lns2] + lns3 + lns8 + [lns6] + lns7
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, fontsize=16, loc="upper right", bbox_to_anchor=(1.4, 1.05))

    plt.show()