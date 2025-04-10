using Random
using LinearAlgebra
using Plots
using JLD2
using DelimitedFiles
using Distributions

##
# Definición de funciones
function potecial(r,k1)
    """
    Definición de la función potencial
    r es la distancia entre las partículas en nm
    k1 es la constante de apantallamiento en nm^-1
    kT es la energía térmica en J y donde la distancia es convertida a nm
    """
    kT = (1.380648e-23*297)*(1e9)^2         # N*m = kg*m^2/s^2 * 
    e = 1.602176565e-19                     # Carga del electrón en C
    ew = 80                                 # Constante dieléctrica del agua
    e0 = 8.8541878176e-12/((1e9)^3)         # Permitividad del vacío C^2/N m^2
    sig = 12.0                              # Tamaño de los pelos de las partículas nm
    k = 1/(k1)                              # longitud de apantallamiento 
    lB = e^2/(kT*4*pi*ew*e0)                # longitu de Bjerrum
    qM = 40*e                               # Número de cargas por partícula
    W(x) = exp(x)/(1+x)                     # Función de apantallamiento
    A = lB*(W(k*sig/2)*qM/e)^2              # Amplitud del potencial
    gu = (A/r)*exp(-k*r)
    return gu
end

function gradiente_numerico_potencial(d_t,k1)
    """
    Función que calcula el gradiente del potencial en función de la distancia
    d_t es la distancia máxima de interacción entre las partículas en nm que para este caso es 3*R
    k1 es la constante de apantallamiento en nm^-1
    """

    M = 200000
    dr = d_t/M

    r = range(0.0,d_t,M)
    bu = zeros(M)

    bu[1] = potecial(r[2],k1)
    for i in 2:M
        bu[i] = potecial(r[i],k1)
    end

    gbu = zeros(M)
    gbu[1] = (bu[3]-bu[1])/(2*dr)
    for i in 2:M-1
        gbu[i] = (bu[i+1]-bu[i-1])/(2*dr)
    end

    return gbu, bu

end

function matriz_fuerza(posiciones, gbu, bu, d_t)
    """
    Esta función genera una matriz cuyos elementos son vectores de fuerza entre las 
    posiciones de la particula i-esima y la j-esima. La matriz es de la forma
    Fij = [Fijx, Fijy, Fijz] donde i y j son las posiciones de las partículas en la esfera
    """

    N = size(posiciones,1)
    F = zeros(size(posiciones,1),size(posiciones,1),3)
    dr = d_t / length(bu)

    for i in 1:N-1

        for j in i+1:N

            local rij, Intr

            rij = posiciones[i,:] - posiciones[j,:]

            producto = dot(posiciones[i,:],posiciones[j,:])/R2

            dij = R*sqrt(2.0-2.0*producto)

            Intr = Int(trunc(dij/dr))

            if Intr == 0
                Intr = 1
            end

            F[i,j,:] = gbu[Intr].*normalize(rij)
            F[j,i,:] = -F[i,j,:]

        end

    end

    return F

end

function potencial_particulas(posiciones, bu, d_t)
    """
    Esta función calcula el potencial entre las partículas en la esfera.
    La función recibe las posiciones de las partículas y el potencial entre ellas 
    y devuelve la suma de los potenciales entre todas las partículas.
    """

    N = size(posiciones,1)
    
    dr = d_t / length(bu)

    V = 0.0

    for i in 1:N-1

        for j in i+1:N

            producto = dot(posiciones[i,:],posiciones[j,:])/R2

            dij = R*sqrt(2.0-2.0*producto)

            Intr = Int(trunc(dij/dr))

            if Intr == 0
                Intr = 1
            end

            V += bu[Intr]

        end

    end

    return V

end

function generate_sphere_points(N,R)
    """
    Función que genera puntos sobre la superficie de una esfera de radio R.
    La función utiliza la fórmula de la espiral de Fibonacci para distribuir los puntos uniformemente.
    """

    points = zeros(N, 3)  # Matriz de puntos (N filas, 3 columnas)

    φ = π * (3.0 - sqrt(5.0))  # Ángulo de oro

    for i in 1:N
        y = 1 - (i - 1) / (N - 1) * 2      # y va de 1 a -1
        radius = sqrt(1 - y^2)             # Radio en el plano xz
        θ = φ * (i - 1)                    # Ángulo de oro incrementado

        x = cos(θ) * radius
        z = sin(θ) * radius

        points[i, :] = [x, y, z]           # Almacenar el punto (x, y, z)
    end

    return R.*points
end

function puntos_ecuador(N,R)
    """
    Función que genera puntos sobre el ecuador de una esfera de radio R.
    La función distribuye los puntos uniformemente a lo largo de la línea del ecuador.
    """

    # Inicializar un arreglo para los puntos en el ecuador
    puntos = zeros(N, 3)
    
    # Calcular ángulos distribuidos uniformemente
    for i in 1:N
        theta = 2 * π * (i - 1) / N  # Ángulo en el plano xy
        x = R * cos(theta)
        y = R * sin(theta)
        puntos[i, :] = [x, y, 0]  # z = 0 en el ecuador
    end
    
    return puntos
end

function puntos_random(N,R)
    """
    Función que genera puntos aleatorios distribuidos uniformemente sobre la superficie de una esfera de radio R.
    """
    
    # Lista de numeros aleatorios con distribucion normal centrada en 0 y son desviacion estandar 1
    ang_rad = zeros(N)
    
    points = zeros(N, 3)

    # Lista de numeros aleatorios con distribucion normal centrada en 0 y son desviacion estandar 1
    y = zeros(N)
    radius = zeros(N) 
    
    for i in range(1,N)
        ang_rad[i] = rand(Uniform{Float16}(-1, 1))
        y[i] = R.*rand(Uniform{Float16}(-1, 1))
        radius[i] = sqrt(1 - y[i]*y[i]/(R^2))
    end
    
    # Es la lista de valores entre -1 y 1 pero se multiplica por pi así se recore toda la esfera
    theta = pi .* ang_rad
    
    # Valores de x y z que en principio son uniformemente aleatorios en toda la supericie
    x = R.*cos.(theta) .* radius
    z = R.*sin.(theta) .* radius

    for n in 1:N
        points[n,:] = [x[n], y[n], z[n]]
    end

    return points
    
end


# ============================================================
# Configuración inicial
# ============================================================


# Definición de algunas variables.
EN = 2                                                  # Configuraciones del sistema
N = 40                                                  # Número de partículas sobre la superficie
k1 = 3*96                                               # Constante de apantallamiento [nm^-1]
R = 100.0                                               # Radio de la esfera [nm]
M = 10000                                               # Número de pasos en el sistema 
Ms = 21                                                 # Subsistemas anidados 
R2 = R^2                                                # Precalcular valores que no cambian dentro de los bucles 
rit = zeros(EN, M, N, 3)                                # Arreglo donde se guardan los datos
D0dt = 0.1                                              # longitud de paso al cuadrado[nm^2]
ki = sqrt(2*D0dt)                                       # Coeficiente para el paso [nm]
Vt = zeros(EN,M)                                        # Vector donde se guarda el potencial  
dis_total = 3*R                                         # Distancia total de interacción
gbu, bu = gradiente_numerico_potencial(dis_total,k1)    # Arreglos de potencial y modulo de fuerza 
ris = zeros(Ms, N, 3)                                   # Arreglo para pasos intermedios
constantes_sim = [EN,N,R,M,Ms,D0dt]
puntos_prueba = puntos_ecuador(N,R)


##
# ============================================================
# Simulación
# ============================================================

tiempo = @elapsed begin

        for en in 1:EN
            global rit, ris, hist, var_t
            local tiempo

            puntos_iniciales = puntos_random(N,R)
            for n in 1:N
                rit[en, 1, n, :] = puntos_iniciales[n,:]        # Asignación de valores iniciales
                ris[1, n, :] = puntos_iniciales[n,:]
            end
            
            # Primer valor del potencial generado por la distribución aleatoria uniforme sobre la superficie
            # de la esfera
            Vt[en,1] = potencial_particulas(puntos_iniciales,bu, dis_total)

            for m in 1:M-1
                local  Ki

                # Generación de los vectores aleatorios Ki
                Ki = ki .* randn(Ms, N, 3)

                for ms in 1:Ms-1
                    local Fij
                    
                    # Calcular la matriz de fuerza entre las partículas
                    Fij = matriz_fuerza( ris[ms, :, :], gbu, bu, dis_total )

                    for n in 1:N
                        local Dxni, dXi, ni, gamma
                        
                        # Calcular Dxni y dXi
                        dXi = 0.0

                        # Matriz de fuerza total
                        FT = sum( -Fij[n,:,:] , dims=1)
                        FT = dropdims(FT,dims=1)
                        
                        # Vector del paso de la partícula
                        Dxni = (1/R2).*cross(ris[ms, n, :], D0dt.*FT + Ki[ms, n, :]  )
                        
                        dXi = norm(Dxni)
                        ni = normalize(Dxni)

                        # Actualizar ris
                        ris[ms + 1, n, :] = cos(dXi) .* ris[ms, n, :] + sin(dXi) .* cross(ni, ris[ms, n, :])
                        
                        # Normalizar el vector y multiplicarlo por R
                        ris[ms + 1, n, :] = R .* normalize(ris[ms + 1, n, :])
                        
                    end

                end

                # reinicio de los ciclos internos y registrar el último paso en la variable rit
                rit[en, m + 1, :, :] = ris[Ms, :, :]
                ris[1, :, :] = ris[Ms, :, :]

                # Guardar el potencial de las partículas en la variable Vt
                Vt[en,m+1] = potencial_particulas(ris[Ms, :, :],bu, dis_total)
                
            end

        end
end

println("Tiempo de ejecución:  $tiempo segundos")


##
# ============================================================
# Guardar resultados
# ============================================================


name = "datos_jld1/datos_N"*string(N)*"/N"*string(N)*"_k"*string(k1)*"_random_M"*string(M)*"_EN"*string(EN)

@save name*".jld2" rit constantes_sim Vt

Vt2 = (1/EN).*dropdims(sum(Vt,dims=1),dims=1)
t = range(1,M)
Vt1 = cat(t,Vt2,dims=2)

writedlm(name*".csv", Vt1, ',')

println("Archivo guardado con el nombre: $name ")
