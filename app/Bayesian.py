# priorbd = Prior base probability of a battery dying = count of days where an outage was inevitable / total days

# bdcount = count of days prior to an outage where the battery was clearly about to suffer an outage

# daycount = total days of data

 

priorbd = bdcount / daycount = 1070 / 54068 based on reviewed data sample = 0.0198

 

# minbdprob = the ultimate minimum probability of a battery outage – this is the base level probability that is always present

 

minbdprob = 0.0014

 

# for each day record the probability that the battery is dying given a measured day-to-day voltage decline normalised for operating mode

# this is based on a straight measure of the number of days this normalised voltage decline has been recorded / the number of times that corresponded with the battery dying

# p_bd_truepos_vd0 is a function of the normalised voltage drop

# p_bd_falsepos_vd0 = 1 - p_bd_truepos_vd0

ptruep.append(p_bd_truepos_vd0)

pfalsep.append(p_bd_falsepos_vd0)

 

# use a variable history length – look at the conditional probability over 1 day, 2 days,…, up to 10 days

for ppi in range(10):

    # pvdbd = probability of normalised voltage drop given battery dying

    pvdbd[ppi].append(np.prod(ptruep[-ppi-1:]))

    # pvd = probability of that normalised voltage drop

    pvd[ppi].append(pvdbd[ppi][-1]* priorbd + np.prod(pfalsep[-ppi-1:])*(1.- priorbd))

    # pbdvd = probability that the battery is dying given the normalised voltage drop

    # P_A|B = P_B|A * P_A / P_B over multiple successive tests

    pbdvd[ppi].append(pvdbd[ppi][-1] * priorbd / pvd[ppi][-1])