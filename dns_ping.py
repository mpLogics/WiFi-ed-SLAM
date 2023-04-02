import subprocess  # For executing a shell command
# from pythonping import ping
import os
import dns.resolver
import time

PING_RESULT = 0
NETWORK_RESULT = 0


#Find current connected AP address
def findDNS():
    dns_resover = dns.resolver.Resolver()
    # print(dns_resover.nameservers[0])
    return dns_resover.nameservers[0]


#Disable network card
def DisableNetwork():
    ''' disable network card '''
    result = os.system(u"netsh interface set interface disable")
    if result == 1:
        print("disable network card failed")
    else:
        print("disable network card successfully")


#Compute min time (assumed direct path) of given AP address (ToF)
def dnsPingMinTime(dns_address, ping_num, interval):
    # if interval == 0:
    #     sys_command = "ping " + dns_address
    #     print(sys_command)
    # else:
    #     sys_command = "ping -i " + str(interval) + " " + dns_address
    #     print(sys_command)
        
    ''' ping '''
    # result = os.system(u"ping -i 0.1 192.168.0.1")  #35.3.121.44    #35.0.18.25
    # result = os.system(sys_command)
    result = subprocess.Popen(
        ['ping', '-c', str(ping_num), '-i', str(interval), dns_address],
        stdout=subprocess.PIPE)
    stdout, stderr = result.communicate()
    output = stdout.decode('ASCII')
    minToF = float(output.split('/')[-4].split()[-1])
    print('Min ToF of ' + str(ping_num) + 'pings for ' + dns_address + ': ' + str(minToF) + 'ms')

    # if result == 0:       #check status
    #     print("GOOD")
    # else:
    #     print("ERROR")
    # return result
    return minToF


#Ping the given AP address
def iterPing(ipAddress, ping_num, interval):        #ping_num: how many pings in one detection; interval: Interval between each ping
    tofList = []

    while True:
        tofList.append(dnsPingMinTime(ipAddress, ping_num, interval))   #### need to create a map with different (AP, ToF, x, y)
        # PING_RESULT = dnsPingMinTime(ipAddress, ping_num, interval)
        # if PING_RESULT == 0:
        #     time.sleep(20)
        # else:
        #     DisableNetwork()
        #     time.sleep(10)


if __name__ == '__main__':
    ip = findDNS()
    iterPing(ip, 5, 0.1)    #ping 5 times as a group, 100ms between each ping (100ms is the minimum interval for Linux command line)

