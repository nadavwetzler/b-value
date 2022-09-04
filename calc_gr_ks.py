import numpy as np

def b_maximum_liklihood(Mv, dm):
    fMeanMag = np.mean(Mv)
    fMinMag = np.min(Mv)
    b = (1/(fMeanMag-(fMinMag-(dm / 2))))*np.log10(np.exp(1))
    return b

def histw(x,bins):
    h = np.zeros(len(x))
    for ii in range(len(bins)):
        h[ii] = sum(x >= bins[ii])
    return h

def moving_average(x,y, n):
    if (n % 2) == 0:
        n=n-1
    if n <= 1:
        n = 3

    buf = int((n-1) / 2)
    pos = np.arange(buf,len(x)-buf)
    ym  = np.zeros(len(pos))
    xm  = np.zeros(len(pos))
    for ii in range(len(xm)):
        ym[ii] = np.mean(y[pos[ii]-buf:pos[ii]+buf])
    xm = x[pos]
    return xm, ym


def calc_b_val(Mv, dm, Mmin, Mmax, KS_thresh = 0.05):
    #---------Parameters ---------------------
    nma = 5 # number of samples for smoothing (b-value & dK-S
    # KS_thresh = 0.05 # the minimum K-S value to set b-value & Mc; If the value is not obtained will serch for minimum
    class b_data():
        pass
    if len(Mv) > 0:
        if Mmax > max(Mv):
            Mmax = max(Mv)
        if Mmin < min(Mv):
            Mmin = min(Mv)
            if Mmin < 0:
                Mmin = 0
        #---------------Set magnitude distribution---------------

        Mm = np.arange(0.1, max(Mv)-dm, dm)
        Mm = np.round(Mm,1)
        Mv = np.round(Mv,1)
        n0 = len(Mm)

        b  = np.zeros(n0)
        a  = np.zeros(n0)
        D  = np.zeros(n0)

        #-----K-S test for each Mc level ------------------------
        for ii in range(n0):
            Mvi = Mv[Mv>=Mm[ii]]
            # Calculate the b-value (maximum likelihood)
            b[ii] = b_maximum_liklihood(Mvi, dm)
            # Make the commulative G-R distribution
            Mmi = Mm[ii::]
            N_obs = np.zeros(len(Mmi))
            for jj in range(len(Mmi)):
                N_obs[jj] = len(Mvi[Mvi>=Mmi[jj]])
            a[ii]    = np.log10(N_obs[0]) + b[ii]*Mmi[0]
            N_model  = 10 ** (a[ii] - b[ii]*Mmi)
            NS_obs   = N_obs / N_obs[0]
            NS_model = N_model / N_model[0]
            D[ii]    = max(abs(NS_obs - NS_model))
            # print('%2.3f, %2.3f, %1.1f' %(D[ii],b[ii],Mm[ii]))

        #------------- moving avg --------------------------------
        Mmn, b = moving_average(Mm,b, nma)
        _, D = moving_average(Mm,D, nma)
        Mm = Mmn

        # import matplotlib.pyplot as plb
        # fig10 = plb.figure(10)
        # ax1 = fig10.add_subplot(2,1,1)
        # ax1.plot(M_smooth, b_smooth)
        # ax1.plot(Mm, b)
        # ax1.set_xlim([2,6])
        # ax1.set_ylim([0,2])
        # ax1.set_title('b-value')
        # ax2 = fig10.add_subplot(2,1,2)
        # ax2.plot(M_smooth, D_smooth)
        # ax2.plot(Mm, D)
        # ax2.set_xlim([2,6])
        # ax2.set_ylim([0,1])
        # ax2.set_title('d K-S')
        # plb.show()

        #------ Limiting minimum search --------------------------------
        I_Mrange = np.logical_and(Mm>=Mmin, Mm <=Mmax)
        Dok  = D[I_Mrange]
        bok  = b[I_Mrange]
        Mmok = Mm[I_Mrange]

        # -------- Searching for threshold or minimum K-S distance -------------
        pos = np.argmin(Dok)
        checkD = 0
        for ii in range(pos):
            if checkD == 0:
                if Dok[ii]<=KS_thresh:
                    pos = ii
                    checkD = 1
                    Dmin = Dok[ii]
        if checkD ==0:
            Dmin = np.min(Dok[Dok >0])
            pos  = np.argmin(Dok[Dok >0])
            print('D obtained above 0.05 (%1.4f)' % Dmin)

        b_data.Dks   = D
        b_data.b     = b
        b_data.Mm    = Mm
        b_data.Mc    = Mmok[pos]
        b_data.b_val = bok[pos]
        b_data.Mv    = Mv
        b_data.dm    = dm
        b_data.KScut = Dmin

        print('b-value %1.3f' % bok[pos])
        print('Mc %1.3f' % Mmok[pos])
    else:
        b_data.Dks   = 0
        b_data.b     = 0
        b_data.Mm    = 0
        b_data.Mc    = 0
        b_data.b_val = 0
        b_data.Mv    = 0
        b_data.dm    = 0
        b_data.KScut = 0


    return b_data


def print_b_val(b_data, M1,M2, ax1, ax2a,name,cBin='gray',cGR='g'):
    MM  = np.arange(0.1,max(b_data.Mv)+0.1,0.1)
    MhR = np.histogram(b_data.Mv, bins=MM)
    MM  = MM[0:-1]
    Mh  = np.zeros(len(MM))

    for jj in range(len(MM)):
        Mh[jj] = np.sum(MhR[0][jj:])

    Mc_p      = np.argmin(np.abs(MM - b_data.Mc))
    MMi       = MM[Mc_p:]
    a_val     = np.log10(Mh[Mc_p]) + b_data.b_val*MMi[0]
    N_model   = 10 ** (a_val - b_data.b_val*MMi)
    N_model_A = 10 ** (a_val - b_data.b_val * MM)


    pow_y = np.ceil(np.log10(np.max(Mh)))
    ax1.scatter(MM, Mh, 30, cGR)
    ax1.scatter(MM, MhR[0], 20, edgecolors=cBin, marker='s', facecolors='none')
    ax1.plot(MM, N_model_A, '--k', linewidth=2)
    ax1.plot(MMi, N_model, '-k', linewidth=2)
    label = r'$log(N)=-%1.2fM + %2.2f$' '\n' r'$M_C=%1.1f$' % (b_data.b_val, a_val, b_data.Mc)
    # ax1.plot([b_data.Mc, b_data.Mc], [1, 10 ** (a_val - b_data.b_val*b_data.Mc)],linestyle='-.',c=cGR,label='Mc %1.1f\nb-value %1.2f' % (b_data.Mc,b_data.b_val))
    ax1.plot([b_data.Mc, b_data.Mc], [1, 10 ** (a_val - b_data.b_val*b_data.Mc)],linestyle='-.',c=cGR,label=label)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('Earthquakes Frequency')
    ax1.set_yscale('log')
    ax1.set_xlim([max([0,np.min(b_data.Mv)]), np.max(b_data.Mv)])
    ax1.set_ylim([1, 1.5*max(Mh)])
    # ax1.legend(title=name,loc='upper right')
    legend1 = ax1.legend(loc='upper right')
    legend1._legend_title_box._children[0]._fontproperties._family = 'Impact'
    legend1._legend_title_box._children[0]._fontproperties.set_size(14)
    legend1.set_title(name)
    ax1.grid()

    alphal = 0.8
    ax2b =  ax2a.twinx()
    I_m = np.logical_and(b_data.Mm >= M1, b_data.Mm <= M2)
    ylim_d = [np.min(b_data.Dks[I_m])-0.05, np.max(b_data.Dks[I_m])]
    ax2a.plot(b_data.Mm, b_data.Dks, linewidth=2, c='g', alpha=alphal)
    ax2a.scatter(b_data.Mc, b_data.KScut, 15, 'k', label='dK-S: %3.2f' % b_data.KScut)
    ax2a.plot([b_data.Mc, b_data.Mc], [0, 2], '-.k')
    ax2a.set_ylabel(r'$\Delta_{KS}$', color='g')
    ax2a.set_ylim(ylim_d)
    ax2a.legend(title=name,loc='upper left')
    ylim_b = [np.min(b_data.b[I_m]), np.max(b_data.b[I_m])]

    ax2b.plot(b_data.Mm,b_data.b, linewidth=2, c='b', alpha=alphal)
    ax2b.set_ylabel('b-value', color='b')
    ax2b.set_xlabel('Magnitude of completness')
    ax2b.set_xlim([max([1,np.min(b_data.Mv)]), M2])
    ax2b.set_ylim(ylim_b)
    # ax2b.legend(title=name,loc='upper left')

    # forceAspect(ax2b,aspect=0.3)
