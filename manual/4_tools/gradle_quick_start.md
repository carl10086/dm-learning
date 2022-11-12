
#tool #gradle

## 1. basic requirements

-   spring integration ;
-   how to set basic properties ;
-   multiple repository ;
-   platform - management of all dependencies ;

**********************components :********************** _build → (1-N) Projects → (1-N) Tasks_

### 1.1 refer

-   [gradle release notes](https://gradle.org/releases/)
-   [Optional dependencies are not optional](https://blog.gradle.org/optional-dependencies)
-   [build script blocks && core types](https://docs.gradle.org/current/dsl/org.gradle.api.Project.html#org.gradle.api.Project:allprojects(groovy.lang.Closure))
-   [maven repository](https://docs.gradle.org/current/userguide/declaring_repositories.html#sec:maven_repo)
-   Jvm Plugins : [library application platform groovy scala .](https://docs.gradle.org/current/userguide/java_library_plugin.html)
-   [spring boot gradle plugin](https://docs.spring.io/spring-boot/docs/current/gradle-plugin/reference/htmlsingle/#introduction)
-   [organizing gradle projects](https://docs.gradle.org/current/userguide/organizing_gradle_projects.html#organizing_gradle_projects)
-   [initialization scripts](https://docs.gradle.org/current/userguide/init_scripts.html)


## 2. tips demo

### 2.1 set pull repos for all projects

`gradle` has a initial script … by [gradle initialization scripts](https://docs.gradle.org/current/userguide/init_scripts.html) .

then we try to [declaring repos](https://docs.gradle.org/current/userguide/declaring_repositories.html) .

```groovy
allprojects {
    repositories {
        mavenLocal()
        maven {
            url '<https://maven.aliyun.com/repository/public>'
        }
        maven {
            credentials {
                username 'Your username'
                password 'Your password'
            }
            url '<https://packages.aliyun.com/maven/repository/..../>'
        }
        maven {
            credentials {
                username 'Your username'
                password 'Your password'
            }
            url '<https://packages.aliyun.com/maven/repository/..../>'
        }
        mavenCentral()
    }
}
```

U can check by

```groovy
tasks.register('showRepos') {
    doLast {
        println "All repos:"
        println repositories.collect { it.name }
    }
}
```

### 2.2 set all publish repos

[just use maven-publish plugin](https://docs.gradle.org/current/userguide/publishing_maven.html) .

```groovy
allprojects {
    apply plugin: "java"
    apply plugin: "maven-publish"
    publishing {
        publications {
            mavenJava(MavenPublication) {
                artifactId = "dm-gd-api"
                from components.java
            }
        }

        repositories {
            maven {
                def releasesRepoUrl = "YOUR RELEASE URL"
                def snapshotsRepoUrl = "YOUR SNAPSHOT URL"
                credentials {
                    username 'YOUR USERNAME'
                    password 'PASSWORD'
                }
                url version.endsWith('SNAPSHOT') ? snapshotsRepoUrl : releasesRepoUrl

            }
        }
    }
} 
```

### 2.3 dependency management with platform .

define with dependencies .

```groovy
plugins {
    id 'java-platform'
    id 'maven-publish'
}

javaPlatform {
    allowDependencies()
}

group 'com.ysz'

dependencies {
    api platform("org.springframework.boot:spring-boot-dependencies:2.7.5")
}

publishing {
    publications {
//        mavenJava(MavenPublication) {
//            artifactId = "dm-platform"
//        }
        maven(MavenPublication) {
//            group = "com.ysz"
//            artifactId = "dm-platform"
            from components.javaPlatform
        }
    }
}
```

use it .

```groovy
dependencies {
//    implementation platform('org.springframework.boot:spring-boot-dependencies:2.7.5
    implementation platform('com.ysz:dm-platform:0.0.0')
 
}
```

### 2.4 understand dependency .

-   implementation
-   compileOnly
-   runtimeOnly

### 2.5 use build Src to abstract imperative logic

good job

### 2.6 lock

```bash
find ~/.gradle -type f -name "*.lock" -delete
```

### 2.7 use java tool chain .

[specify jvm versions](https://docs.gradle.org/current/userguide/toolchains.html#toolchains)

### 2.8 migrating build logic from groovy to kotlin

****before u start****

-   use latest version of gradle
-   only support idea and android studio
-   there are some situations where the Kotlin DSL is slower
-   java 8 or higher .
-   …