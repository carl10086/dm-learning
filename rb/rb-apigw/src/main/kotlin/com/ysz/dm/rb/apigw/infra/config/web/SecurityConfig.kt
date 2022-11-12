package com.ysz.dm.rb.apigw.infra.config.web

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.http.HttpMethod
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.core.AuthenticationException
import org.springframework.security.core.userdetails.User
import org.springframework.security.core.userdetails.UserDetailsService
import org.springframework.security.provisioning.InMemoryUserDetailsManager
import org.springframework.security.web.SecurityFilterChain
import org.springframework.security.web.authentication.www.BasicAuthenticationEntryPoint
import org.springframework.security.web.util.matcher.AntPathRequestMatcher
import org.springframework.security.web.util.matcher.RequestMatcher
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/12
 **/
@Configuration
@EnableWebSecurity(debug = false)
open class SecurityConfig {

    @Bean
    open fun customAuthenticationEntryPoint(): CustomAuthenticationEntryPoint {
        return CustomAuthenticationEntryPoint()
    }


    @Bean
    open fun filterChain(http: HttpSecurity): SecurityFilterChain {
        http
            .csrf().disable()
            .authorizeRequests()
            .requestMatchers(
                CustomAuthenticationMatcher(
                    /*direct allow if patterns matches*/
                    directorAllowPatterns = listOf(
                        AntPathRequestMatcher("/index/hello")
                    )
                )
            )
            .authenticated()
            /*just let it go*/
            .anyRequest()
            .permitAll()
            .and()
            .httpBasic()
            .authenticationEntryPoint(customAuthenticationEntryPoint())
        return http.build()
    }

    /**
     * in memory user details manager
     */
    @Bean
    open fun userDetailsService(): UserDetailsService? {
        val user = User.withDefaultPasswordEncoder().username("user").password("password").roles("USER").build()
        return InMemoryUserDetailsManager(user)
    }


    class CustomAuthenticationMatcher(
        /*directory allow, don't care methods*/
        private var directorAllowPatterns: List<AntPathRequestMatcher> = emptyList(),
    ) : RequestMatcher {


        /**
         * @return true: means reject, false means pass
         */
        override fun matches(request: HttpServletRequest): Boolean {

            directorAllowPatterns.firstOrNull { x -> x.matches(request) }?.let {
                return false
            }

            var httpMethod = HttpMethod.valueOf(request.method)


            return when (httpMethod) {
                HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH -> true
                else -> false
            }
        }
    }


    class CustomAuthenticationEntryPoint : BasicAuthenticationEntryPoint() {

        override fun afterPropertiesSet() {
            realmName = "Realm-gw"
            super.afterPropertiesSet()
        }

        override fun commence(
            request: HttpServletRequest,
            response: HttpServletResponse,
            authException: AuthenticationException
        ) {
            response.addHeader("WWW-Authenticate", "Basic realm=\"" + this.realmName + "\"")
            /*401*/
            response.status = HttpServletResponse.SC_UNAUTHORIZED
            /*write json ?*/
            response.contentType = "application/json"
            response.characterEncoding = "UTF-8"
        }
    }

}