package com.ysz.dm.arm.client.decorators

import com.linecorp.armeria.client.ClientRequestContext
import com.linecorp.armeria.client.HttpClient
import com.linecorp.armeria.client.SimpleDecoratingHttpClient
import com.linecorp.armeria.common.HttpRequest
import com.linecorp.armeria.common.HttpResponse
import com.linecorp.armeria.common.logging.RequestLogProperty
import org.slf4j.LoggerFactory
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-28 6:13 PM
 */
class CustomHttpDecorator(delegate: HttpClient) : SimpleDecoratingHttpClient(delegate) {


    companion object {
        private val log = LoggerFactory.getLogger(CustomHttpDecorator::class.java)

        fun newDecorator(): Function<in HttpClient, CustomHttpDecorator> {
            return Function { delegate: HttpClient ->
                CustomHttpDecorator(delegate)
            }
        }
    }

    override fun execute(ctx: ClientRequestContext, req: HttpRequest): HttpResponse {
        log.info("before execute")
        val params = "custom params"

        ctx.log().whenAvailable(
            RequestLogProperty.REQUEST_START_TIME,
            RequestLogProperty.REQUEST_HEADERS,
            RequestLogProperty.NAME,
            RequestLogProperty.SESSION
        ).thenAccept {
            val ctx = it.context()
            log.info("on request:{}", params)

            ctx.log().whenComplete().thenAccept {
                log.info("on response:{}", params)
            }
        }
        return this.unwrap().execute(ctx, req)
    }
}