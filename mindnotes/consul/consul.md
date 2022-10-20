## basic consul server with security

- a service networking solution that enables you to manage secure network connectivity ;

By the end of this tutorial, you will have deployed a Consul Server agent running on the extra virtual machine .

In order to securely configure consul by ensuring all communications between the Consul Server and clients are
inaccessible to unintended agents .

- A gossip encryption key .
- A root certificate authority certificate from a private CA .

**1) interact with consul server**

```bash
$ consul members
    Node    Address          Status  Type    Build   Protocol  DC   Partition  Segment
    consul  10.2.31.35:8301  alive   server  1.12.4  2         dc1  default    <all>ux
```

**2) interact with consul kv**

```
consul kv put consul/configuration/db_port 5432
```

**3) interact with consul dns**

```bash
$ dig @127.0.0.1 -p 8600 consul.service.consul

; <<>> DiG 9.16.27-Debian <<>> @127.0.0.1 -p 8600 consul.service.consul
; (1 server found)
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 48200
;; flags: qr aa rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 4096
;; QUESTION SECTION:
;consul.service.consul.        IN    A

;; ANSWER SECTION:
consul.service.consul.    0    IN    A    172.20.0.3

;; Query time: 0 msec
;; SERVER: 127.0.0.1#8600(127.0.0.1)
;; WHEN: Tue Aug 23 15:45:53 UTC 2022
;; MSG SIZE  rcvd: 66
```

- use Consul's KV store as a centralized configuration
- consul server as a DNS server.

In the next, you will deploy consul clients on the VMs hosting your application. Then , you will register the services
running on each server and set up health checks for each services . This enables service discovery using Consul's
distributed health check system and DNS .

## register services

By the end of this tutorial, you will have deployed and started a consul client agent on each VMs that hosts . In
addition, you will have registered the services in the consul service catalog and setup health check for each service .

```
$ tree ./client_configs   
./client_configs
|-- agent-client-secure.hcl
|-- api
|   |-- agent-client-acl-tokens.hcl
|   |-- agent-client-secure.hcl
|   |-- agent-gossip-encryption.hcl
|   |-- consul-agent-ca.pem
|   `-- svc-api.hcl
|-- db
|   |-- agent-client-acl-tokens.hcl
|   |-- agent-client-secure.hcl
|   |-- agent-gossip-encryption.hcl
|   |-- consul-agent-ca.pem
|   `-- svc-db.hcl
|-- frontend
|   |-- agent-client-acl-tokens.hcl
|   |-- agent-client-secure.hcl
|   |-- agent-gossip-encryption.hcl
|   |-- consul-agent-ca.pem
|   `-- svc-frontend.hcl
`-- nginx
    |-- agent-client-acl-tokens.hcl
    |-- agent-client-secure.hcl
    |-- agent-gossip-encryption.hcl
    |-- consul-agent-ca.pem
    `-- svc-nginx.hcl

4 directories, 21 files
```

copy configuration on client VMS .

First, create the directories to Consul in all client nodes . The command will create both the configuration dir and the
data dir on the client nodes .



members listed as belows:

```
$ consul members
Node                Address           Status  Type    Build   Protocol  DC   Partition  Segment
consul              10.2.19.238:8301  alive   server  1.12.4  2         dc1  default    <all>
hashicups-api       10.2.2.165:8301   alive   client  1.12.4  2         dc1  default    <default>
hashicups-db        10.2.17.106:8301  alive   client  1.12.4  2         dc1  default    <default>
hashicups-frontend  10.2.4.60:8301    alive   client  1.12.4  2         dc1  default    <default>
hashicups-nginx     10.2.14.142:8301  alive   client  1.12.4  2         dc1  default    <default>
```

**1) query services**

```
$ consul catalog services -tags
api           v1
consul        
db            v1
frontend      v1
nginx         v1
```


Next, create the new configuration file, Notice that this configuration adds a `v2`  tag to the database service .

```
## svc-db.hcl
service {
  name = "db"
  id = "db-1"
  tags = ["v1", "v2"]
  port = 5432
 
  check {
    id =  "check-db",
    name = "Product db status check",
    service_id = "db-1",
    tcp  = "localhost:5432",
    interval = "1s",
    timeout = "1s"
  }
}
```

then you have to reload your service definition .

```
$ consul reload
Configuration reload triggered
```

```
$ consul catalog services -tags
api           v1
consul        
db            v1,v2
frontend      v1
nginx         v1
```