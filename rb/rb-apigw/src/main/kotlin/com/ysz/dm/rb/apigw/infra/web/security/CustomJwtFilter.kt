package com.ysz.dm.rb.apigw.infra.web.security

import com.auth0.jwt.JWT
import com.auth0.jwt.algorithms.Algorithm
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.core.userdetails.User
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource
import org.springframework.web.filter.OncePerRequestFilter
import java.time.Instant
import javax.servlet.FilterChain
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

/**
 *<pre>
 * custom jwt filter .
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/14
 **/
class CustomJwtFilter : OncePerRequestFilter() {
    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        chain: FilterChain
    ) {
        var header = request.getHeader(HttpHeaders.AUTHORIZATION)

        /*1. extract jwt bearer header */
        if (header?.startsWith(BEARER) != true) {
            chain.doFilter(request, response)
            return
        }

        /*2. token*/
        val token = header.substring(BEARER.length)
        var payload = verify(token)

        if (null == payload) {
            chain.doFilter(request, response)
            return
        } else {
            val authUser = User.withUsername(payload.subject).password("1234567").roles(Roles.ADMIN).build()
            val authentication = UsernamePasswordAuthenticationToken(
                authUser, "", authUser.authorities
            )

            authentication.details = WebAuthenticationDetailsSource().buildDetails(request)

            SecurityContextHolder.getContext().authentication = authentication

            chain.doFilter(request, response)
        }
    }


    companion object {
        private val log = LoggerFactory.getLogger(CustomJwtFilter::class.java)

        const val BEARER = "Bearer "

        private val sign = Algorithm.HMAC256("123456")

        private fun verify(token: String): PayLoad? {
            return try {
                val decode = JWT.require(sign).build().verify(token)

                PayLoad(
                    decode.issuedAt.toInstant(), decode.expiresAt.toInstant(), decode.subject
                )
            } catch (ignored: Exception) {
                null
            }
        }
    }


    private data class PayLoad(
        val issueAt: Instant, val expireAt: Instant, val subject: String
    )


}