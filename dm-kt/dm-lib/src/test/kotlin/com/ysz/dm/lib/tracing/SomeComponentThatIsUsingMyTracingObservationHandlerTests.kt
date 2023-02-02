package com.ysz.dm.lib.tracing

import io.micrometer.observation.transport.Kind
import io.micrometer.observation.transport.RequestReplySenderContext
import io.micrometer.tracing.Tracer
import io.micrometer.tracing.handler.TracingObservationHandler

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/2/2
 **/
internal class SomeComponentThatIsUsingMyTracingObservationHandlerTests {
}

data class CustomReq(val reqId: String)
data class CustomResp(val resp: String)

data class CustomContext() : RequestReplySenderContext<CustomReq, CustomResp>({ _, _, _ ->

}, Kind.CLIENT)


class MyTracingObservationHandler : TracingObservationHandler<CustomContext> {
    override fun getTracer(): Tracer {
        TODO("Not yet implemented")
    }
}

