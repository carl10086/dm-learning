## intro


refer:

- [why java record are better](https://nipafx.dev/java-record-semantics/)


# 4S-UC033 📦⚙️ Request an access token for enrollment

**Sequence diagram:**

```plantuml
@startuml


participant "eID App" as eid
boundary "Enrollment\nOAuth authorization server" as idras
participant "Core Platform" as cp

eid -> idras : request access token for enrollment
activate idras

idras -> cp : RegisterSecurityPrincipal
activate cp
activate idras
cp -> cp : register security principal\nand create session
activate cp
deactivate cp
cp -->> idras : sessionId
deactivate cp

deactivate idras
idras -->> eid : access token for enrollment (sessionId)
deactivate idras

@enduml
