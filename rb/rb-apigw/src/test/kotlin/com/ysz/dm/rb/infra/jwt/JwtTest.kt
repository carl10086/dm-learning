package com.ysz.dm.rb.infra.jwt

import com.auth0.jwt.JWT
import com.auth0.jwt.algorithms.Algorithm
import com.ysz.dm.rb.base.core.tools.id.IdTools
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import java.time.Duration
import java.time.Instant
import java.util.*

/**
 * @author carl
 * @create 2022-11-14 1:44 PM
 **/
internal class JwtTest {

    @Test
    internal fun `test_createThenVerifyJwt`() {
        val now = Instant.now()
        val token = JWT.create()
            .withIssuedAt(now)
            .withSubject("user001")
            .withExpiresAt(Date(now.toEpochMilli() + Duration.ofDays(1L).toMillis()))
            .withJWTId(IdTools.uuid())
            .withClaim("custom", "")
            .sign(hmaC256)

        log.info("sign:${token}")
        log.info("verify result:{}", verify(token))
    }

    private fun verify(token: String): PayLoad? {
        return try {
            val decode = JWT.require(hmaC256).build().verify(token)

            PayLoad(
                decode.issuedAt.toInstant(), decode.expiresAt.toInstant(), decode.subject, decode.id
            )
        } catch (ignored: Exception) {
            null
        }
    }

    companion object {
        private val hmaC256 = Algorithm.HMAC256("123456")
        private val log = LoggerFactory.getLogger(JwtTest::class.java)
    }


    data class PayLoad(
        val issueAt: Instant, val expireAt: Instant, val subject: String, val jwtId: String
    )
}