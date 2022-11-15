package com.ysz.dm.rb.user.infra.persist.dao

import com.ysz.dm.rb.user.domain.user.UserAuthStatus
import com.ysz.dm.rb.user.domain.user.UserRegisterComeFrom
import com.ysz.dm.rb.user.infra.persist.dataobject.UserCoreDO
import com.ysz.dm.rb.user.infra.testcfg.H2JpaConfig
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import org.springframework.context.annotation.AnnotationConfigApplicationContext

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
internal class UserDaoTest {
    private val ctx = AnnotationConfigApplicationContext(H2JpaConfig::class.java)
    var userDao: UserDao = ctx.getBean(UserDao::class.java)


    @BeforeEach
    internal fun setUp() {
        userDao.save(mockUser(1L).apply { username = "aaaa" })
        userDao.save(mockUser(2L).apply { username = "bbbb" })
    }


    @Test
    internal fun `test_findAll`() {
        userDao.findAll().forEach {
            log.info("x:{}", it)
        }
    }


    companion object {
        private val log = LoggerFactory.getLogger(UserDaoTest::class.java)

        private fun mockUser(id: Long, now: Long = System.currentTimeMillis()) = UserCoreDO(
            id,
            UserRegisterComeFrom.RB_ORIGINAL_WEB.code,
            "",
            now,
            now,
            "",
            "",
            "",
            UserAuthStatus.AUDITING.code,
            now,
            now
        )
    }
}