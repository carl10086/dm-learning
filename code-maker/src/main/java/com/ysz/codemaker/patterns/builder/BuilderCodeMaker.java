package com.ysz.codemaker.patterns.builder;

import com.github.mustachejava.DefaultMustacheFactory;
import com.github.mustachejava.Mustache;
import com.github.mustachejava.MustacheFactory;
import com.google.common.collect.Lists;
import com.ysz.codemaker.patterns.builder.core.Cfg;
import com.ysz.codemaker.patterns.builder.core.Context;
import com.ysz.codemaker.patterns.builder.core.Field;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * @author carl
 * @create 2022-10-12 4:24 PM
 **/
public class BuilderCodeMaker {

  private MustacheFactory factory = new DefaultMustacheFactory();

  public String execute(Cfg cfg) throws Exception {
    Mustache m = factory.compile("tpl/builder/object.mustache");

    StringWriter writer = new StringWriter();

    List<Field> fields = Lists.newArrayList(
        Field.ofArray("httpHosts", "org.apache.http.HttpHost.HttpHost"),
        Field.ofSimple("threadNum", "java.lang.Integer"),
        Field.ofSimple("connectTimeoutMills", "java.lang.Integer"),
        Field.ofSimple("socketTimeoutMills", "java.lang.Integer"),
        Field.ofSimple("ioThreadCnt", "java.lang.Integer")
    );

    List<String> imports = new ArrayList<>();
    for (Field field : fields) {
      String s = field.fullClassname();
      if (!s.startsWith("java.lang")) {
        imports.add(field.fullClassname());
      }
    }

    m.execute(writer, Lists.newArrayList(new Context("CustomEsCfg", fields, imports)));

    return writer.toString();
  }

}
