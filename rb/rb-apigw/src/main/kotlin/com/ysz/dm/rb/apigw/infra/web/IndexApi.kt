package com.ysz.dm.rb.apigw.infra.web

import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/12
 **/
@RestController
@RequestMapping("/index")
class IndexApi {


    @GetMapping("/hello")
    fun hello() = "hello security"


    @PostMapping("/post")
    fun post() = "post"
}