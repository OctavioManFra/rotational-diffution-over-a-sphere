using Random
using LinearAlgebra
using Plots
using JLD2
using DelimitedFiles
using Distributions
using StatsBase

function puntos_random(N,R)
    
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



# Definición de algunas variables.
EN = 1                                      # Configuraciones del sistema
N = 20000                                   # Número de partículas sobre la superficie
R = 100.0                                   # Radio de la esfera [nm]
M = 200                                     # Número de pasos en el sistema 
Ms = 21                                     # Subsistemas anidados 
R2 = R^2                                    # Precalcular valores que no cambian dentro de los bucles 
rit = zeros(EN, M, N, 3)                    # Arreglo donde se guardan los datos
ris = zeros(Ms, N, 3)                       # Arreglo para pasos intermedios
D0dt = 10   
ki = sqrt(2*D0dt)                           # Coeficiente para el paso [nm]
Vt = zeros(EN,M)                            # Vector donde se guarda el potencial
zta_p =101                                  # Divisiones en el ángulo azimutal
var_tiempo = [10,30,50,200]                 # Vector de tiempos
histograma = zeros(zta_p-1, length(var_tiempo))
bins = range(-1, 1, zta_p)
ang_t = zeros(EN,M*(Ms-1))
constantes_sim = [EN,N,R,M,Ms,D0dt]



for en in 1:EN

    tiempo = @elapsed begin

        global rit, ris, hist, var_tiempo
        local tiempo
        
        for n in 1:N
            rit[en, 1, n, :] = [0,0,R]       # Asignación de valores iniciales
            ris[1, n, :] = [0,0,R]
        end

        for m in 1:M-1

            local  Ki

            # Generación de los vectores aleatorios Ki
            Ki = ki .* randn(Ms, N, 3)

            for ms in 1:Ms-1

                for n in 1:N

                    local Dxni, dXi, ni, gamma
                    
                    # Calcular Dxni y dXi

                    Dxni = (1/R2).*cross(ris[ms, n, :], Ki[ms, n, :])
                    
                    dXi = norm(Dxni)

                    ni = normalize(Dxni)

                    # Actualizar ris
                    ris[ms + 1, n, :] = cos(dXi) .* ris[ms, n, :] + sin(dXi) .* cross(ni, ris[ms, n, :])
                    
                    # Normalizar el vector y multiplicarlo por R
                    ris[ms + 1, n, :] = R .* normalize(ris[ms + 1, n, :])
                    
                end

            end
            rit[en, m + 1, :, :] = ris[Ms, :, :]
            ris[1, :, :] = ris[Ms, :, :]
            
        end
    end

    println("Tiempo de ejecución: ", tiempo, " segundos, en la configuracion $(en)")

    # Creación de histograma de la ultima posición de la particula
    for t in eachindex(var_tiempo)
        local hist

        hist = fit(Histogram, Float64[], bins)  # Inicializar histograma vacío

        for n in 1:N

            local zta

            # Calcular tta
            zta = rit[en,var_tiempo[t],n,3]/R

            # Actualizar el histograma
            push!(  hist  ,  clamp( zta , -1.0 , 1.0 - (2/(zta_p)) )  )
            
        end

        histograma[:,t] += hist.weights

    end 

end

##

A = 2*N/R

##
 
histograma_particulas = histograma./A


##


tiempo_polinomios = @elapsed begin

    mpx = 100000              # Numero de pasos en [-1, 0] y [0, 1]

    npx = 2*mpx+1             # Numero de pasos totales 2*n+1 para incluir el 0 

    nlg = 200                 # Orden de los polinomios

    P = zeros((nlg,npx))      # Se define una matriz de polinomios de dimensión (N, npx)

    x = range(-1,1,npx)       # Vector de posición

    dx = 2/npx                # Tamaño de paso

    P[1,:] .= 1.0             # Polinomio de orden P_0[x]

    P[2,:] .= x               # Polinomio de orden P_1[x]

    # Ciclo para aumentar el orden del polinomio

    for n in 1:nlg-2

        P[n+2,:] = ( ( 2 * n + 1 ) .* x .* P[n+1,:] - n .* P[n,:] ) / ( n + 1 )

    end

end
println("Tiempo de cálculo de los polinomios: $tiempo_polinomios")

##

l_max = 120

tiempo_suma_polinomios = @elapsed begin

    suma_poli = zeros(npx,length(var_tiempo))

    for t in eachindex(var_tiempo)

        for l in 1:l_max 
            suma_poli[:,t] .+= ( (2*(l-1) + 1)/(2) )*exp(-((l-1)*l)*D0dt*(Ms-1)*var_tiempo[t]/R2).*P[l,:]
        end
    end

end


##
theta_h = range(0,180,zta_p)
theta_p = range(0,180,npx)

grafica = plot()
grafica = plot!(theta_p, reverse(suma_poli))
grafica = plot!(theta_h[2:end],  histograma_particulas, xlabel = "zta", ylabel = "Frecuencia", title = "Histograma de la posición final de las partículas",ylims=[-0.1,2])
display(grafica)
#savefig(grafica,"distribucion_particula_libre.png")

#print(size(angulos))
#angulos_avg = (1/EN).*dropdims(  sum(angulos,dims=1)  , dims=1 )
#ang = plot()
#ang = plot!( range(1,(Ms-1)*M), ang_t[1,:] )
#println(sum(ang_t[1,:])/(length(ang_t[1,:])-1))
#ang = ylims!(-0.005,2.5)
#display(ang)
#println(angulos[2,2,:])

##

println(size(histograma))
println(size(bins))


for t in eachindex(var_tiempo)

    polinomios = cat(theta_p, reverse(suma_poli[:,t]),dims=2)
    writedlm("particula_libre/polinomios$(var_tiempo[t]).csv", polinomios , ',')

    histograma = cat(theta_h[2:end], histograma_particulas[:,t],dims=2)
    writedlm("particula_libre/histograma$(var_tiempo[t]).csv", histograma , ',')

end

