import numpy as np
import matplotlib.pyplot as plt
from .functions import Q_func, M_func

def UBs(i, t, X, Y, U, B, V, control_params, type_plot='imshow', forward=False):
    print("t: ", t[i])
    rows, cols = X.shape

    # Get sample for quiver
    s_r, s_c = rows // int(0.1 * rows), cols // int(0.1 * cols)
    X_s, Y_s = X[::s_r, ::s_c], Y[::s_r, ::s_c]

    # Reconstruct the active spray footprint only for forward plots.
    spray_outline = None
    if forward and control_params is not None:
        plot_time = t[i]
        control_vars = np.asarray(control_params.get("control_vars", []))
        duration = control_params.get("T")

        if control_vars.size and duration is not None:
            control_vars = control_vars.reshape(-1, 4)
            start_times = control_vars[:, 3]

            active_spray = np.any(
                (start_times <= plot_time)
                & (plot_time <= start_times + duration)
            )

            if active_spray:
                Q = Q_func(X, Y, plot_time, control_params)
                M = M_func(X, Y, plot_time, control_params)

                # Q and M have the same spray footprint, but this also works
                # if either suppression coefficient is zero.
                spray_outline = np.maximum(np.abs(Q), np.abs(M))

                if np.max(spray_outline) == 0:
                    spray_outline = None

    def add_spray_outline():
        if spray_outline is not None:
            plt.contour(
                X,
                Y,
                spray_outline,
                levels=[0.1 * np.max(spray_outline)],
                colors='black',
                linestyles=':',
                linewidths=1.5,
            )

    # Figure size
    plt.figure(figsize=(10, 4))

    # Temperature plot
    plt.subplot(1, 2, 1)

    if i >= 0:
        U = U[i]
        B = B[i]

    if type_plot == 'contour':
        temp = plt.contourf(
            X, Y, U, cmap=plt.cm.jet, alpha=0.8,
            vmin=np.min(U), vmax=np.max(U)
        )
    elif type_plot == 'pcolor':
        temp = plt.pcolor(
            X, Y, U, cmap=plt.cm.jet, alpha=0.8,
            vmin=np.min(U), vmax=np.max(U)
        )
    elif type_plot == 'imshow':
        temp = plt.imshow(
            U, origin='lower', cmap=plt.cm.jet, alpha=0.8,
            vmin=np.min(U), vmax=np.max(U),
            extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]]
        )
    else:
        raise Exception("Type of plot not defined.")

    if forward:
        add_spray_outline()

    if V is not None:
        if type(V) is np.ndarray:
            if len(V[0].shape) > 2 and len(V[1].shape) > 2:
                V1, V2 = V[i, 0, ::s_r, ::s_c], V[i, 1, ::s_r, ::s_c]
            else:
                V1, V2 = V[i, 0], V[i, 1]
        else:
            V1, V2 = V(X_s, Y_s, t[i])

        print(
            "V1: [%f, %f], V2: [%f, %f]"
            % (np.min(V1), np.max(V1), np.min(V2), np.max(V2))
        )
        plt.quiver(X_s, Y_s, V1, V2)

    cb1 = plt.colorbar(temp, fraction=0.046, pad=0.04)
    cb1.set_label("Temperature", size=14)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    # Fuel plot
    plt.subplot(1, 2, 2)
    fuel = plt.pcolor(X, Y, B, cmap=plt.cm.Oranges)

    if forward:
        add_spray_outline()

    cb2 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
    cb2.set_label("Fuel Fraction", size=14)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    plt.tight_layout()
    plt.show()

"""
def UBs(i, t, X, Y, U, B, V, control_params, type_plot='imshow', forward=False):
    print("t: ", t[i])
    rows, cols = X.shape
    # Get sample for quiver
    s_r, s_c =  rows // int(0.1 * rows), cols // int(0.1 * cols)
    X_s, Y_s = X[::s_r,::s_c], Y[::s_r,::s_c]
    # Figure size
    plt.figure(figsize=(10, 4)) 
    # Left plot
    plt.subplot(1, 2, 1)
    # Index -1 is used for last plot option
    if i >= 0:
        U = U[i]
        B = B[i]
    if type_plot == 'contour':
        temp = plt.contourf(X, Y, U, cmap=plt.cm.jet, alpha=0.8, vmin=np.min(U), vmax=np.max(U))
    elif type_plot == 'pcolor':
        temp = plt.pcolor(X, Y, U, cmap=plt.cm.jet, alpha=0.8, vmin=np.min(U), vmax=np.max(U))
    elif type_plot == 'imshow':
        temp = plt.imshow(U, origin='lower', cmap=plt.cm.jet, alpha=0.8,  
            vmin=np.min(U), vmax=np.max(U), extent=[X[-1, 0], X[-1, -1], Y[0, -1], Y[-1, -1]])
    else:
        raise Exception("Type of plot not defined.")
    
    if V is not None:
        if type(V) is np.ndarray:
            if len(V[0].shape) > 2 and len(V[1].shape) > 2: # Check shape of vector field array
                V1, V2 = V[i, 0, ::s_r, ::s_c], V[i, 1, ::s_r, ::s_c]
            else:
                V1, V2 = V[i, 0], V[i, 1]
        else:
            V1, V2 = V(X_s, Y_s, t[i])
        #print(V1, V2)
        print("V1: [%f, %f], V2: [%f, %f]" % (np.min(V1), np.max(V1), np.min(V2), np.max(V2)))
        plt.quiver(X_s, Y_s, V1, V2)
        #plt.quiver(X_s, Y_s, V[0](X_s, Y_s, t[i]), V[1](X_s, Y_s, t[i]))
    cb1 = plt.colorbar(temp, fraction=0.046, pad=0.04)
    cb1.set_label("Temperature", size=14)
    #plt.title("Temperature and wind")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.subplot(1, 2, 2)
    fuel = plt.pcolor(X, Y, B, cmap=plt.cm.Oranges)
    cb2 = plt.colorbar(fuel, fraction=0.046, pad=0.04)
    cb2.set_label("Fuel Fraction", size=14)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()
    plt.show()

def UB(t, X, Y, U, B, V):
    UBs(-1, t, X, Y, U, B, V)
"""