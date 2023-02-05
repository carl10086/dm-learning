package com.ysz.dm.lib.tracing

import io.micrometer.observation.transport.Kind
import io.micrometer.observation.transport.Propagator
import io.micrometer.observation.transport.RequestReplySenderContext
import io.micrometer.tracing.Tracer
import io.micrometer.tracing.handler.TracingObservationHandler
import io.micrometer.tracing.test.simple.SimpleTracer

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

data class CustomContext(
    val othersSetter: Propagator.Setter<CustomReq> = Propagator.Setter { _, _, _ -> }
) :
    RequestReplySenderContext<CustomReq, CustomResp>(
        othersSetter, Kind.CLIENT
    )


class MyTracingObservationHandler(private val childTracer: Tracer) : TracingObservationHandler<CustomContext> {
    override fun getTracer(): Tracer = childTracer
}


fun main() {
    val tracer = SimpleTracer()
    val newSpan = tracer.nextSpan().name("calculateTax")

    tracer.withSpan(newSpan.start()).use {
        try {
            newSpan.tag("taxValue", "111")
            newSpan.event("taxCalculated")
        } finally {
            newSpan.end()
        }
    }

    val ctx = tracer.currentTraceContext()
    println(ctx)
}