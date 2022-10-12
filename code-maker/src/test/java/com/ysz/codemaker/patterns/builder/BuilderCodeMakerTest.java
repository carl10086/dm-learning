package com.ysz.codemaker.patterns.builder;

import com.ysz.codemaker.patterns.builder.core.Cfg;
import org.junit.Before;
import org.junit.Test;

/**
 * @author carl
 * @create 2022-10-12 4:27 PM
 **/
public class BuilderCodeMakerTest {

  private BuilderCodeMaker codeMaker;

  @Before
  public void setUp() throws Exception {
    this.codeMaker = new BuilderCodeMaker();
  }

  @Test
  public void execute() throws Exception {
    System.out.println(this.codeMaker.execute(new Cfg()));
  }
}