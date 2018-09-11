//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
package com.konakartadmin.apiexamples;

import com.konakartadmin.appif.KKAdminIf;

/**
 * This class shows how to call the "Custom" interface of the KonaKart Admin API.
 * 
 * This is used as an example to demonstrate how to customise the engine code by providing a custom
 * implementation of the "Custom" interface.
 * 
 * Before running you may have to change the username and password used to log into the engine. The
 * default values are admin@konakart.com / princess .
 */
public class CustomExamples extends BaseApiExample
{
    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            CustomExamples test = new CustomExamples();

            /**
             * In this first run we use the default engine - note that this will not pick up any
             * customisations so we should see a null returned from the custom() call because that's
             * the default implementation
             */
            test.testUsingEngine("com.konakartadmin.bl.KKAdmin");

            /**
             * In this second run we instantiate the custom admin engine. This will in turn pick up
             * our engine customisations - and hence we should see a different result from our call
             * to custom()
             */
            test.testUsingEngine("com.konakartadmin.app.KKAdminCustomEng");

            /**
             * In this third run we instantiate the admin Web Services client engine. This will send
             * our custom() call over SOAP to a running KonaKart system (identified by the URL in
             * konakartadmin_axis_client.properties). The engine that is instantiated by that remote
             * system running the web services will instantiate the engine identified in the
             * konakartadmin.properties file by the konakart.admin.ws.engine.classname property
             */
            test.testUsingEngine("com.konakartadmin.ws.KKWSAdmin");
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private void testUsingEngine(String engineClassName) throws Exception
    {
        /*
         * Instantiate a KonaKart Admin Engine instance
         */
        Class<?> engineClass = Class.forName(engineClassName);
        eng = (KKAdminIf) engineClass.newInstance();

        /*
         * Login with default credentials
         */
        sessionId = eng.login(DEFAULT_USERNAME, DEFAULT_PASSWORD);

        System.out.println("\nCall the custom interface 'Custom' using Engine named "
                + engineClassName);

        String result = eng.custom("input1", "input2");

        System.out.println("Result was: " + result + "\n");
    }
}
