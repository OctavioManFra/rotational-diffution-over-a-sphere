
using JLD2
using Distributed
using DelimitedFiles
using Plots
addprocs()
@everywhere using LinearAlgebra
@everywhere using LegendrePolynomials  
@everywhere using StatsBase




##
###########  Sección para crear los polinomios de legendre  ########## 


tiempo_polinomios = @elapsed begin

    mpx = 100000              # Numero de pasos en [-1, 0] y [0, 1]

    npx = 2*mpx+1             # Numero de pasos totales 2*n+1 para incluir el 0 

    nlg = 300                 # Orden de los polinomios

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
###########  Función que calcula por 3 metrodos la función de distribución radial


@everywhere function g_theta_parallel(l_max, taus, constantes_sim, zta_p, R2, M_in)

    EN, N, M, M_start = Int(constantes_sim[1]), Int(constantes_sim[2]), Int(constantes_sim[4]), Int(constantes_sim[4]) - M_in
    
    #dzt1 = 2 / zta_p
    #dzt2 = 2 / npx 
    EN_temp = EN
    is = 1:N-1
    js = 1:N
    zta_bins = range(0, 180, zta_p + 1)
    zta1 = (zta_bins[1:end-1] .+ zta_bins[2:end]) ./ 2
    zta2 = range(0, 180, npx)
    tau_max = 1000

    #zta_bins = zeros(zta_p)
    #zta_var = range(0, 180-(2/(zta_p - 1)) , zta_p)
    #
    #for j in 1:zta_p-1
    #    zta_bins[j] = ( (2*j - zta_p) * (1/zta_p) + ( 2*(j+1) - zta_p )* (1/zta_p) )/2
    #end

    function coeficientes_integrados(funcion, l_max)
        # Inicializar variables
        Polinomios = zeros(l_max, zta_p)
        coeficientes = zeros(l_max)
        bins = range(-1, 1, zta_p + 1)
        posiciones = (bins[1:end-1] .+ bins[2:end]) ./ 2
        for x in eachindex(posiciones)
            Polinomios[:,x] = collect(collectPl(posiciones[x], lmax = l_max - 1))
        end
    
        # Calcular coeficientes
        for l in 1:l_max
            coeficientes[l] = ((2 * (l - 1) + 1) / 2) * dot(funcion, Polinomios[l, :])
        end
    
        return coeficientes
    end

    function process_diff_dots(i, ris, rjs)
        # Inicializar variables
        bins = range(-1, 1, zta_p + 1)
        hist_diff = fit(Histogram, Float64[], bins)     # Histograma vacío
        hl1_temp_diff = zeros(1, l_max)                 # Coeficientes temporales
        epsilon = 1e-14                                 # Tolerancia para evitar errores numéricos
    
        # Calcular productos punto para partículas diferentes
        for j in i+1:N
            dot_product = dot(ris[i, :], rjs[j, :]) / R2
    
            if -1.0 - epsilon < dot_product < 1.0 + epsilon
                dot_product = clamp(dot_product, -1.0, 1.0)     # Asegurar que esté en el rango [-1, 1]
                push!(hist_diff, dot_product)                   # Agregar al histograma
                hl1_temp_diff[1, :] .+= 2 .* collect(collectPl(dot_product, lmax = l_max - 1))
            end
        end
    
        return hl1_temp_diff, hist_diff.weights
    end

    function process_self_dots(j, ris, rjs)
        # Inicializar variables
        bins = range(-1, 1, zta_p + 1)
        hl1_temp_self = zeros(1, l_max)             # Coeficientes temporales
        hist_self = fit(Histogram, Float64[], bins) # Histograma vacío
        epsilon = 1e-14                             # Tolerancia para errores numéricos

        # Calcular producto punto para partículas iguales
        dot_product = dot(ris[j, :], rjs[j, :]) / R2

        if -1.0 - epsilon < dot_product < 1.0 + epsilon
            dot_product = clamp(dot_product, -1.0, 1.0)     # Asegurar que esté en el rango [-1, 1]
            push!(hist_self, dot_product)                   # Agregar al histograma
            hl1_temp_self[1, :] .+= collect(collectPl(dot_product, lmax = l_max - 1))
        end

        return hl1_temp_self, hist_self.weights
    end

    for t in eachindex(taus)

        local tau, coeficint, polinomio, histogram, coef_diff, coef_self , poli_diff, poli_self, hist_diff, hist_self
        local hl_diff, hl_self, g_theta_diff, g_theta_self, fl_diff, fl_self, g_theta

        hl_diff = zeros(1, l_max)
        hl_self = zeros(1, l_max)
        fl_diff = zeros(1, l_max)
        fl_self = zeros(1, l_max)
        g_theta_diff = zeros(zta_p)
        g_theta_self = zeros(zta_p)
        hl = zeros(1, l_max)
        g_theta = zeros(zta_p)

        tau = taus[t]
        
        println("tau = $(taus[t])")

        for en in 1:EN

            for m in M_start : M - tau_max

                local partial_results1, partial_results2, ris, rjs

                # Sumar los resultados parciales de `process_ij_dots`

                ris = rit[en, m + tau, :, :]
                rjs = rit[en, m , :, :]

                partial_results1 = pmap(i -> process_diff_dots(i, ris, rjs), is)

                for res in partial_results1

                    hl_diff[:, :] += res[1]
                    g_theta_diff[:] += res[2]

                end
                
                partial_results1 = nothing  # Liberar memoria

                partial_results2 = pmap(j -> process_self_dots(j, ris, rjs), js)

                for res in partial_results2

                    hl_self[:, :] += res[1]
                    g_theta_self[:] += res[2]

                end

                partial_results2 = nothing  # Liberar memoria

            end

        end

        # Normalizar resultados

        hl_diff[:, :] /= ( ( M - M_start - tau_max + 1) * EN_temp )
        hl_self[:, :] /= ( ( M - M_start - tau_max + 1) * EN_temp )
        hl[:, :] = hl_diff[:, :] + hl_self[:, :]

        fl_diff[1, :] = hl_diff[1, :] 
        fl_self[1, :] = hl_self[1, :] 
        
        hl_diff[1, :] .+= N
        hl_self[1, :] .+= N
        hl[:,:] .+= N
        
        hl[1, 1] = (hl[1, 1] .- N) / (N^2) - 1
        hl_diff[1, 1] = (hl_diff[1, 1] .- N) / (N^2) - 1
        hl_self[1, 1] = (hl_self[1, 1] .- N) / (N^2) - 1

        for l in 2:l_max
            
            hl[1, l] = (2 * (l - 1) + 1) * ((hl[1, l] .- N) / (N^2))
            hl_diff[1, l] = (2 * (l - 1) + 1) * ((hl_diff[1, l] - N) / (N^2))
            hl_self[1, l] = (2 * (l - 1) + 1) * ((hl_self[1, l] - N) / (N^2))
        
        end
        
        A1 = 1 / ((1 / (zta_p - 1)) * (N^2) * (M - M_start - tau_max + 1) * EN_temp)
        A2 = 2 / ((1 / (zta_p - 1)) * (N^2) * (M - M_start - tau_max + 1) * EN_temp)
        
        g_theta_self[:] = reverse(g_theta_self[:]) .* A1
        g_theta_diff[:] = reverse(g_theta_diff[:]) .* A2
        
        g_theta[:] = g_theta_self[:] .+ g_theta_diff[:]
        
        polinomio = cat(zta2, reverse(dropdims(hl[:, :] * P[1:l_max, :], dims=1) .+ 1.), dims=2)
        poli_diff = cat(zta2, reverse(dropdims(hl_diff[:, :] * P[1:l_max, :], dims=1) .+ 1.), dims=2)
        poli_self = cat(zta2, reverse(dropdims(hl_self[:, :] * P[1:l_max, :], dims=1) .+ 1.), dims=2)
        
        integrado = cat(range(1, l_max), coeficientes_integrados(g_theta[:], l_max), dims=2)
        inte_self = cat(range(1, l_max), coeficientes_integrados(g_theta_self[:], l_max), dims=2)
        inte_diff = cat(range(1, l_max), coeficientes_integrados(g_theta_diff[:], l_max), dims=2)
        
        coeficint = cat(range(1, l_max), hl[1, :], dims=2)
        coef_diff = cat(range(1, l_max), hl_diff[1, :], dims=2)
        coef_self = cat(range(1, l_max), hl_self[1, :], dims=2)
        
        histogram = cat(zta1, g_theta[:], dims=2)
        hist_diff = cat(zta1, g_theta_diff[:], dims=2)
        hist_self = cat(zta1, g_theta_self[:], dims=2)
        
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"polinomio$(tau_var[t]).csv", polinomio , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"poli_diff$(tau_var[t]).csv", poli_diff , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"poli_self$(tau_var[t]).csv", poli_self , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"integrado$(tau_var[t]).csv", integrado , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"inte_self$(tau_var[t]).csv", inte_self , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"inte_diff$(tau_var[t]).csv", inte_diff , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"coeficnts$(tau_var[t]).csv", coeficint , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"coef_diff$(tau_var[t]).csv", coef_diff , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"coef_self$(tau_var[t]).csv", coef_self , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"histogram$(tau_var[t]).csv", histogram , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"hist_diff$(tau_var[t]).csv", hist_diff , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"hist_self$(tau_var[t]).csv", hist_self , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"fl_self_t$(tau_var[t]).csv", cat(range(1,l_max) , fl_self[1,:] , dims = 2 ) , ',')
        writedlm("datos_jld1/datos_N"*string(N)*"/resultado_N"*string(N)*"/resultado_k"*string(k1)*"fl_diff_t$(tau_var[t]).csv", cat(range(1,l_max) , fl_diff[1,:] , dims = 2 ) , ',')

    end


end


 
##
###########  Cargar y definir los parámetros del histograma y el caso de simulacion  #########

N = 40

k1 = 3*96

EN = 2

M = 10000

zta_p = 400

tau_var = [0]

l_max = 120
 
M_inicio = 9000

name = "datos_jld1/datos_N"*string(N)*"/N"*string(N)*"_k"*string(k1)*"_random_M"*string(M)*"_EN"*string(EN)*".jld2"

@load name rit constantes_sim

R , M , Ms , D0dt , phi_p, R2 =  constantes_sim[3],  Int(constantes_sim[4]), Int(constantes_sim[5]), constantes_sim[6], 8, constantes_sim[3]^2

##########  Ejecución del código  ########## 

tiempo_total = @elapsed begin

    g_theta_parallel(l_max,tau_var,constantes_sim,zta_p,R2,M_inicio)

end

println("Tiempo total de ejecución: $tiempo_total")
