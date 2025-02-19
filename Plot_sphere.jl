using Random
using Distributions
using PlotlyJS
#using Plots

function puntos_random(N, R)
    points = zeros(N, 3)

    for i in 1:N
        # Distribución uniforme en la esfera
        θ = 2π * rand()
        φ = acos(2 * rand() - 1)

        # Coordenadas cartesianas
        x = R * sin(φ) * cos(θ)
        y = R * sin(φ) * sin(θ)
        z = R * cos(φ)

        points[i, :] = [x, y, z]
    end

    return points
end

N, R = 100, 100

points = puntos_random(N, R)

# Parametrización de la esfera
u = range(0, 2π, length=50)
v = range(0, π, length=50)
    
X = [R * sin(vj) * cos(ui) for ui in u, vj in v]
Y = [R * sin(vj) * sin(ui) for ui in u, vj in v]
Z = [R * cos(vj) for ui in u, vj in v]

esfera = surface(z=Z, x=X, y=Y, showscale=true, colorscale=nothing, opacity = 0.25, surfacecolor=fill("rgba(0, 150, 129, 0.2)", size(X)))
puntos = scatter3d(x=points[:,1], y=points[:,2], z=points[:,3], mode="markers", markersize=2, color=:red, label="Puntos en la esfera")

# Ajustar la cámara
layout_settings = Layout(
    scene=attr(
        camera=attr(
            eye=attr(x=1.5, y=1.5, z=1.5)
        ),
        xaxis=attr(
            title="X",
            titlefont=attr(size=30),
            tickfont=attr(size=22),
            gridcolor="black",  # Líneas del eje X en negro
            zerolinecolor="black",  # Línea cero del eje X en negro
            showgrid=true,  # Mostrar cuadrícula
            showline=true,  # Mostrar línea del eje
            linecolor="black",  # Color de la línea del eje X
        ),
        yaxis=attr(
            title="Y",
            titlefont=attr(size=30),
            tickfont=attr(size=22),
            gridcolor="black",  # Líneas del eje Y en negro
            zerolinecolor="black",  # Línea cero del eje Y en negro
            showgrid=true,  # Mostrar cuadrícula
            showline=true,  # Mostrar línea del eje
            linecolor="black",  # Color de la línea del eje Y
        ),
        zaxis=attr(
            title="Z",
            titlefont=attr(size=30),
            tickfont=attr(size=22),
            gridcolor="black",  # Líneas del eje Z en negro
            zerolinecolor="black",  # Línea cero del eje Z en negro
            showgrid=true,  # Mostrar cuadrícula
            showline=true,  # Mostrar línea del eje
            linecolor="black",  # Color de la línea del eje Z
        )
    )
)


# Crear la figura con PlotlyJS
p = PlotlyJS.plot([
    esfera,  # Superficie semitransparente
    puntos  # Puntos en la esfera
], layout_settings)

# Guardar como imagen (PNG)
PlotlyJS.savefig(p, "esfera.png",width=3840, height=2160, scale=3)