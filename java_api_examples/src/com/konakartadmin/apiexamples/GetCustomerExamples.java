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

import java.net.URL;
import java.util.Iterator;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;

import com.konakart.app.KKException;
import com.konakart.util.KKConstants;
import com.konakart.util.PropertyFileFinder;
import com.konakartadmin.app.AdminCustomer;
import com.konakartadmin.app.AdminEngineConfig;
import com.konakartadmin.app.KKAdminException;
import com.konakartadmin.ws.KKAdminEngineMgr;

/**
 * This class shows how to call a custom version of the "getCustomerForId" interface of the KonaKart
 * Admin API.
 * 
 * Before running you may have to change the username and password used to log into the engine. The
 * default values are admin@konakart.com / princess .
 */
public class GetCustomerExamples extends BaseApiExample
{
    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            GetCustomerExamples test = new GetCustomerExamples();

            /**
             * In this first run we use the default engine - note that this will not pick up any
             * customisations so we should see a null returned in the custom5 attribute of the
             * selected customer.
             */
            test.testUsingEngine("com.konakartadmin.bl.KKAdmin", null);

            /**
             * In this second run we instantiate the custom admin engine. This will in turn pick up
             * our engine customisations - and hence we should see a different result in the custom5
             * attribute of the selected customer.
             */
            test.testUsingEngine("com.konakartadmin.app.KKAdminCustomEng", null);

            /**
             * In this third run we instantiate the admin Web Services client engine. This will send
             * our custom() call over SOAP to a running KonaKart system (identified by the URL in
             * konakartadmin_axis_client.properties). The engine that is instantiated by that remote
             * system running the web services will instantiate the engine identified in the
             * konakartadmin.properties file by the konakart.admin.ws.engine.classname property
             */
            test.testUsingEngine("com.konakartadmin.ws.KKWSAdmin", null);

            /**
             * In this fourth run we instantiate the admin Web Services client engine again but this
             * time using the properties array to specify the URL of the web service. Again this
             * will send our custom() call over SOAP to a running KonaKart system (identified by the
             * URL in konakartadmin_axis_client.properties). The engine that is instantiated by that
             * remote system running the web services will instantiate the engine identified in the
             * konakartadmin.properties file by the konakart.admin.ws.engine.classname property
             */
            test.testUsingEngine("com.konakartadmin.ws.KKWSAdmin", test.getPropertiesFromFile(
                    KKConstants.KONAKARTADMIN_WS_CLIENT_PROPERTIES_FILE, null));

            /**
             * In this fifth run we instantiate the admin RMI Services client engine using the
             * standard properties file. Again this will send our custom() call over RMI to a
             * running KonaKart system (identified by the konakart.rmi.host and konakart.rmi.port
             * properties in konakartadmin.properties). The engine that is instantiated by that
             * remote system running the web services will instantiate the engine identified in the
             * konakartadmin.properties file by the konakart.admin.rmi.engine.classname property
             */
            test.testUsingEngine("com.konakartadmin.rmi.KKRMIAdminEng", null);

            /**
             * In this sixth run we instantiate the admin RMI Services client engine again but this
             * time using the properties array to specify all the properties we need from the
             * konakartadmin.properties file. We don't need all of the properties for RMI so we
             * restrict the selection to those beginning with "konakart.rmi". The other properties
             * in the file aren't used so can be ignored. Again this will send our custom() call
             * over RMI to a running KonaKart system (identified by the konakart.rmi.host and
             * konakart.rmi.port properties in konakartadmin.properties). The engine that is
             * instantiated by that remote system running the web services will instantiate the
             * engine identified in the konakartadmin.properties file by the
             * konakart.admin.rmi.engine.classname property
             */
            test.testUsingEngine("com.konakartadmin.rmi.KKRMIAdminEng", test.getPropertiesFromFile(
                    KKConstants.KONAKARTADMIN_PROPERTIES_FILE, "konakart.rmi"));
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private String[] getPropertiesFromFile(String fileName, String keysStartingWith)
            throws KKAdminException, KKException, ConfigurationException
    {
        /*
         * Find the specified properties file which is guaranteed to return the URL of
         * the properties file or throw an exception.
         */
        URL configFileURL = PropertyFileFinder.findPropertiesURL(fileName);

        PropertiesConfiguration conf = new PropertiesConfiguration(configFileURL);

        if (conf.isEmpty())
        {
            throw new KKAdminException("The configuration file: " + configFileURL
                    + " does not appear to contain any keys");
        }

        // Now create the array of properties

        Iterator<?> keys = (conf.getKeys());
        int keyCount = 0;
        while (keys.hasNext())
        {
            String key = (String) keys.next();
            String value = (String) conf.getProperty(key);
            if (keysStartingWith == null || key.startsWith(keysStartingWith))
            {
                System.out.println(keyCount + ") " + key + " => " + value);
                keyCount++;
            }
        }

        String[] properties = new String[keyCount * 2];
        keys = (conf.getKeys());
        int propIdx = 0;
        while (keys.hasNext())
        {
            String key = (String) keys.next();
            String value = (String) conf.getProperty(key);
            if (keysStartingWith == null || key.startsWith(keysStartingWith))
            {
                properties[propIdx++] = key;
                properties[propIdx++] = value;
            }
        }

        return properties;
    }

    private void testUsingEngine(String engineClassName, String[] properties) throws Exception
    {
        /*
         * Instantiate a KonaKart Admin Engine instance by name
         */
        KKAdminEngineMgr kkAdminEngMgr = new KKAdminEngineMgr();
        AdminEngineConfig adEngConf = new AdminEngineConfig();
        adEngConf.setMode(AdminEngineConfig.MODE_SINGLE_STORE);
        adEngConf.setPropertiesFileName(KKConstants.KONAKARTADMIN_PROPERTIES_FILE);
        adEngConf.setAxisClientFileName(KKConstants.KONAKARTADMIN_WS_CLIENT_PROPERTIES_FILE);
        adEngConf.setStoreId(getStoreId());
        adEngConf.setCustomersShared(isCustomersShared());
        adEngConf.setProductsShared(isProductsShared());
        adEngConf.setProperties(properties);

        try
        {
            /*
             * This creates a KonaKart Admin Engine by name using the constructor that requires an
             * AdminEngineConfig object. This is the recommended approach.
             */
            setEng(kkAdminEngMgr.getKKAdminByName(engineClassName, adEngConf));
        } catch (java.lang.ClassNotFoundException cnf)
        {
            System.out.println("\nCould not find class: " + engineClassName);
            if (engineClassName.equals("com.konakartadmin.rmi.KKRMIAdminEng"))
            {
                System.out.println(engineClassName + " is only provided in the Enterprise version");
            }
            System.out.println();
            return;
        }

        /*
         * Login with default credentials
         */
        sessionId = eng.login(DEFAULT_USERNAME, DEFAULT_PASSWORD);

        System.out.println("\nCall 'getCustomerForId' using Engine named " + engineClassName);

        /*
         * You may have to modify the customer Id in this call to find a customer in your database!
         */
        AdminCustomer cust = eng.getCustomerForId(sessionId, 1);

        if (cust == null)
        {
            System.out.println("No Customer Found with that Id");
        } else
        {
            System.out.println("Found: " + cust.getFirstName() + " " + cust.getLastName());
            System.out.println("custom5 = " + cust.getCustom5() + "\n");
        }

    }
}
